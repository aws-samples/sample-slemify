// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

// Package k8s provides a client-go wrapper for Kubernetes API interaction:
// create/apply manifests, watch Job status, stream pod logs, check Deployment readiness.
package k8s

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	s3types "github.com/aws/aws-sdk-go-v2/service/s3/types"

	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	policyv1 "k8s.io/api/policy/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	sigsyaml "sigs.k8s.io/yaml"
)

// Client wraps client-go for Slemify operations.
type Client struct {
	clientset     kubernetes.Interface
	dynamicClient dynamic.Interface
	namespace     string
}

// NewClient creates a K8s client from kubeconfig path.
// If kubeconfigPath is empty, uses default loading rules (~/.kube/config or in-cluster).
func NewClient(kubeconfigPath, namespace string) (*Client, error) {
	var config *rest.Config
	var err error

	if kubeconfigPath != "" {
		config, err = clientcmd.BuildConfigFromFlags("", kubeconfigPath)
	} else {
		loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
		configOverrides := &clientcmd.ConfigOverrides{}
		config, err = clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
			loadingRules, configOverrides).ClientConfig()
	}
	if err != nil {
		return nil, fmt.Errorf("building kubeconfig: %w", err)
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("creating kubernetes client: %w", err)
	}

	dynClient, err := dynamic.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("creating dynamic client: %w", err)
	}

	return &Client{
		clientset:     clientset,
		dynamicClient: dynClient,
		namespace:     namespace,
	}, nil
}

// EnsureNamespace creates the namespace if it doesn't exist.
func (c *Client) EnsureNamespace(ctx context.Context) error {
	ns := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: c.namespace,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by": "slemify",
			},
		},
	}

	_, err := c.clientset.CoreV1().Namespaces().Create(ctx, ns, metav1.CreateOptions{})
	if err != nil && !errors.IsAlreadyExists(err) {
		return fmt.Errorf("creating namespace %s: %w", c.namespace, err)
	}
	return nil
}

// ApplyManifest applies an unstructured manifest using server-side apply semantics.
// Falls back to create-or-update for resources that don't support SSA.
func (c *Client) ApplyManifest(ctx context.Context, obj *unstructured.Unstructured) error {
	gvr := gvrFromUnstructured(obj)

	// Determine if resource is cluster-scoped or namespaced
	var resourceClient dynamic.ResourceInterface
	if isClusterScoped(obj.GetKind()) {
		resourceClient = c.dynamicClient.Resource(gvr)
	} else {
		resourceClient = c.dynamicClient.Resource(gvr).Namespace(c.namespace)
	}

	// Retry loop for optimistic concurrency conflicts
	for attempt := 0; attempt < 5; attempt++ {
		// Try to get existing resource
		existing, err := resourceClient.Get(ctx, obj.GetName(), metav1.GetOptions{})
		if err != nil {
			if !errors.IsNotFound(err) {
				return fmt.Errorf("checking existing resource: %w", err)
			}
			// Create
			_, err = resourceClient.Create(ctx, obj, metav1.CreateOptions{})
			if err != nil {
				return fmt.Errorf("creating %s/%s: %w", obj.GetKind(), obj.GetName(), err)
			}
			return nil
		}

		// Update — preserve resourceVersion for optimistic concurrency
		obj.SetResourceVersion(existing.GetResourceVersion())
		_, err = resourceClient.Update(ctx, obj, metav1.UpdateOptions{})
		if err != nil {
			if errors.IsConflict(err) && attempt < 4 {
				time.Sleep(time.Duration(attempt+1) * 500 * time.Millisecond)
				continue
			}
			return fmt.Errorf("updating %s/%s: %w", obj.GetKind(), obj.GetName(), err)
		}
		return nil
	}
	return fmt.Errorf("updating %s/%s: max retries exceeded", obj.GetKind(), obj.GetName())
}

// isClusterScoped returns true for known cluster-scoped resource kinds.
func isClusterScoped(kind string) bool {
	clusterScoped := map[string]bool{
		"Namespace":          true,
		"NodePool":           true,
		"EC2NodeClass":       true,
		"NodeOverlay":        true,
		"PersistentVolume":   true,
		"ClusterRole":        true,
		"ClusterRoleBinding": true,
	}
	return clusterScoped[kind]
}

// SubmitJob creates a Job and returns its name.
// If a Job with the same name already exists, it is deleted first
// (along with its dependent pods) to allow a clean re-run.
func (c *Client) SubmitJob(ctx context.Context, job *batchv1.Job) (string, error) {
	if job.Namespace == "" {
		job.Namespace = c.namespace
	}

	// Clean up existing Job if present (from a previous failed run)
	existing, err := c.clientset.BatchV1().Jobs(job.Namespace).Get(ctx, job.Name, metav1.GetOptions{})
	if err == nil && existing != nil {
		propagation := metav1.DeletePropagationBackground
		_ = c.clientset.BatchV1().Jobs(job.Namespace).Delete(ctx, job.Name, metav1.DeleteOptions{
			PropagationPolicy: &propagation,
		})
		// Wait briefly for the old Job to be cleaned up
		for i := 0; i < 10; i++ {
			_, err := c.clientset.BatchV1().Jobs(job.Namespace).Get(ctx, job.Name, metav1.GetOptions{})
			if errors.IsNotFound(err) {
				break
			}
			time.Sleep(time.Second)
		}
	}

	created, err := c.clientset.BatchV1().Jobs(job.Namespace).Create(ctx, job, metav1.CreateOptions{})
	if err != nil {
		return "", fmt.Errorf("submitting job: %w", err)
	}
	return created.Name, nil
}

// WatchJobUntilDone watches a Job until it succeeds, fails, or the context is cancelled.
// If the watch channel closes (e.g., Spot interruption), it polls the Job status before failing.
func (c *Client) WatchJobUntilDone(ctx context.Context, jobName string) error {
	for {
		watcher, err := c.clientset.BatchV1().Jobs(c.namespace).Watch(ctx, metav1.ListOptions{
			FieldSelector: fmt.Sprintf("metadata.name=%s", jobName),
		})
		if err != nil {
			return fmt.Errorf("watching job %s: %w", jobName, err)
		}

		result := c.watchLoop(ctx, watcher, jobName)
		watcher.Stop()

		if result == nil {
			return nil // Job completed
		}

		// On watch errors, check the Job status directly before giving up
		if isWatchError(result) {
			status, pollErr := c.checkJobStatus(ctx, jobName)
			if pollErr != nil {
				return result
			}
			if status == "Complete" {
				return nil
			}
			if status == "Failed" {
				return fmt.Errorf("job %s failed", jobName)
			}
			// Job still running — reconnect the watch
			fmt.Printf("  Watch reconnecting (job still running)...\n")
			continue
		}

		return result
	}
}

func (c *Client) watchLoop(ctx context.Context, watcher watch.Interface, jobName string) error {
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case event, ok := <-watcher.ResultChan():
			if !ok {
				return fmt.Errorf("watch channel closed for job %s", jobName)
			}
			if event.Type == watch.Error {
				return fmt.Errorf("watch error for job %s", jobName)
			}

			job, ok := event.Object.(*batchv1.Job)
			if !ok {
				continue
			}

			for _, cond := range job.Status.Conditions {
				switch cond.Type {
				case batchv1.JobComplete:
					if cond.Status == corev1.ConditionTrue {
						return nil
					}
				case batchv1.JobFailed:
					if cond.Status == corev1.ConditionTrue {
						return fmt.Errorf("job %s failed: %s", jobName, cond.Message)
					}
				}
			}
		}
	}
}

func (c *Client) checkJobStatus(ctx context.Context, jobName string) (string, error) {
	job, err := c.clientset.BatchV1().Jobs(c.namespace).Get(ctx, jobName, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	for _, cond := range job.Status.Conditions {
		if cond.Type == batchv1.JobComplete && cond.Status == corev1.ConditionTrue {
			return "Complete", nil
		}
		if cond.Type == batchv1.JobFailed && cond.Status == corev1.ConditionTrue {
			return "Failed", nil
		}
	}
	return "Running", nil
}

func isWatchError(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	return strings.Contains(msg, "channel closed") ||
		strings.Contains(msg, "connection refused") ||
		strings.Contains(msg, "i/o timeout")
}

// WaitForDeploymentReady waits until a Deployment has all replicas ready.
func (c *Client) WaitForDeploymentReady(ctx context.Context, name string, timeout time.Duration) error {
	deadline := time.After(timeout)
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-deadline:
			return fmt.Errorf("timeout waiting for deployment %s to be ready", name)
		case <-ticker.C:
			dep, err := c.clientset.AppsV1().Deployments(c.namespace).Get(ctx, name, metav1.GetOptions{})
			if err != nil {
				continue // retry on transient errors
			}
			if dep.Status.ReadyReplicas >= *dep.Spec.Replicas {
				return nil
			}
		}
	}
}

// StreamPodLogs streams logs from the first pod matching the given label selector.
// If follow is true, the stream stays open for new log lines.
func (c *Client) StreamPodLogs(ctx context.Context, labelSelector string, follow bool, w io.Writer) error {
	pods, err := c.clientset.CoreV1().Pods(c.namespace).List(ctx, metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err != nil {
		return fmt.Errorf("listing pods: %w", err)
	}
	if len(pods.Items) == 0 {
		return fmt.Errorf("no pods found matching selector %q", labelSelector)
	}

	podName := pods.Items[0].Name
	req := c.clientset.CoreV1().Pods(c.namespace).GetLogs(podName, &corev1.PodLogOptions{
		Follow: follow,
	})

	stream, err := req.Stream(ctx)
	if err != nil {
		return fmt.Errorf("streaming logs from pod %s: %w", podName, err)
	}
	defer stream.Close()

	_, err = io.Copy(w, stream)
	return err
}

// GetJobPodLogs retrieves logs from the pod(s) of a completed or failed Job.
func (c *Client) GetJobPodLogs(ctx context.Context, jobName string) (string, error) {
	selector := fmt.Sprintf("job-name=%s", jobName)
	pods, err := c.clientset.CoreV1().Pods(c.namespace).List(ctx, metav1.ListOptions{
		LabelSelector: selector,
	})
	if err != nil {
		return "", fmt.Errorf("listing pods for job %s: %w", jobName, err)
	}
	if len(pods.Items) == 0 {
		return "", fmt.Errorf("no pods found for job %s", jobName)
	}

	podName := pods.Items[0].Name
	req := c.clientset.CoreV1().Pods(c.namespace).GetLogs(podName, &corev1.PodLogOptions{})
	result, err := req.Do(ctx).Raw()
	if err != nil {
		return "", fmt.Errorf("getting logs from pod %s: %w", podName, err)
	}
	return string(result), nil
}

// gvrFromUnstructured extracts the GroupVersionResource from an unstructured object.
func gvrFromUnstructured(obj *unstructured.Unstructured) schema.GroupVersionResource {
	gvk := obj.GroupVersionKind()

	// Simple pluralization — covers the common K8s resource types we generate.
	plural := pluralize(gvk.Kind)

	return schema.GroupVersionResource{
		Group:    gvk.Group,
		Version:  gvk.Version,
		Resource: plural,
	}
}

// pluralize converts a Kind to its plural resource name (lowercase).
func pluralize(kind string) string {
	known := map[string]string{
		"Namespace":           "namespaces",
		"ConfigMap":           "configmaps",
		"Job":                 "jobs",
		"Deployment":          "deployments",
		"Service":             "services",
		"PodDisruptionBudget": "poddisruptionbudgets",
		"NodePool":            "nodepools",
		"ScaledObject":        "scaledobjects",
		"EC2NodeClass":        "ec2nodeclasses",
	}
	if p, ok := known[kind]; ok {
		return p
	}
	// Fallback: lowercase + "s"
	return fmt.Sprintf("%ss", toLower(kind))
}

// toLower converts a string to lowercase.
func toLower(s string) string {
	if s == "" {
		return s
	}
	b := []byte(s)
	for i, c := range b {
		if c >= 'A' && c <= 'Z' {
			b[i] = c + 32
		}
	}
	return string(b)
}

// ApplyDeployment creates or updates a Deployment.
func (c *Client) ApplyDeployment(ctx context.Context, dep *appsv1.Deployment) error {
	if dep.Namespace == "" {
		dep.Namespace = c.namespace
	}
	existing, err := c.clientset.AppsV1().Deployments(dep.Namespace).Get(ctx, dep.Name, metav1.GetOptions{})
	if err != nil {
		if !errors.IsNotFound(err) {
			return fmt.Errorf("checking deployment %s: %w", dep.Name, err)
		}
		_, err = c.clientset.AppsV1().Deployments(dep.Namespace).Create(ctx, dep, metav1.CreateOptions{})
		return err
	}
	dep.ResourceVersion = existing.ResourceVersion
	_, err = c.clientset.AppsV1().Deployments(dep.Namespace).Update(ctx, dep, metav1.UpdateOptions{})
	return err
}

// ApplyService creates or updates a Service.
func (c *Client) ApplyService(ctx context.Context, svc *corev1.Service) error {
	if svc.Namespace == "" {
		svc.Namespace = c.namespace
	}
	existing, err := c.clientset.CoreV1().Services(svc.Namespace).Get(ctx, svc.Name, metav1.GetOptions{})
	if err != nil {
		if !errors.IsNotFound(err) {
			return fmt.Errorf("checking service %s: %w", svc.Name, err)
		}
		_, err = c.clientset.CoreV1().Services(svc.Namespace).Create(ctx, svc, metav1.CreateOptions{})
		return err
	}
	svc.ResourceVersion = existing.ResourceVersion
	_, err = c.clientset.CoreV1().Services(svc.Namespace).Update(ctx, svc, metav1.UpdateOptions{})
	return err
}

// ApplyPDB creates or updates a PodDisruptionBudget.
func (c *Client) ApplyPDB(ctx context.Context, pdb *policyv1.PodDisruptionBudget) error {
	if pdb.Namespace == "" {
		pdb.Namespace = c.namespace
	}
	existing, err := c.clientset.PolicyV1().PodDisruptionBudgets(pdb.Namespace).Get(ctx, pdb.Name, metav1.GetOptions{})
	if err != nil {
		if !errors.IsNotFound(err) {
			return fmt.Errorf("checking PDB %s: %w", pdb.Name, err)
		}
		_, err = c.clientset.PolicyV1().PodDisruptionBudgets(pdb.Namespace).Create(ctx, pdb, metav1.CreateOptions{})
		return err
	}
	pdb.ResourceVersion = existing.ResourceVersion
	_, err = c.clientset.PolicyV1().PodDisruptionBudgets(pdb.Namespace).Update(ctx, pdb, metav1.UpdateOptions{})
	return err
}

// ApplyYAML parses raw YAML and applies it as an unstructured resource.
func (c *Client) ApplyYAML(ctx context.Context, yamlData []byte) error {
	obj := &unstructured.Unstructured{}
	if err := sigsyaml.Unmarshal(yamlData, &obj.Object); err != nil {
		return fmt.Errorf("parsing YAML: %w", err)
	}
	return c.ApplyManifest(ctx, obj)
}

// ApplyConfigMap creates or updates a ConfigMap.
func (c *Client) ApplyConfigMap(ctx context.Context, cm *corev1.ConfigMap) error {
	if cm.Namespace == "" {
		cm.Namespace = c.namespace
	}
	existing, err := c.clientset.CoreV1().ConfigMaps(cm.Namespace).Get(ctx, cm.Name, metav1.GetOptions{})
	if err != nil {
		if !errors.IsNotFound(err) {
			return fmt.Errorf("checking configmap %s: %w", cm.Name, err)
		}
		_, err = c.clientset.CoreV1().ConfigMaps(cm.Namespace).Create(ctx, cm, metav1.CreateOptions{})
		return err
	}
	cm.ResourceVersion = existing.ResourceVersion
	_, err = c.clientset.CoreV1().ConfigMaps(cm.Namespace).Update(ctx, cm, metav1.UpdateOptions{})
	return err
}

// DetectNodeRole reads the role from the default EC2NodeClass in the cluster.
// Falls back to "KarpenterNodeRole-<clusterName>" if not found.
func (c *Client) DetectNodeRole(ctx context.Context, clusterName string) string {
	gvr := schema.GroupVersionResource{
		Group:    "karpenter.k8s.aws",
		Version:  "v1",
		Resource: "ec2nodeclasses",
	}
	obj, err := c.dynamicClient.Resource(gvr).Get(ctx, "default", metav1.GetOptions{})
	if err == nil {
		spec, ok := obj.Object["spec"].(map[string]interface{})
		if ok {
			if role, ok := spec["role"].(string); ok && role != "" {
				return role
			}
		}
	}
	return fmt.Sprintf("KarpenterNodeRole-%s", clusterName)
}

// UploadToS3 uploads data to an S3 bucket.
func (c *Client) UploadToS3(ctx context.Context, bucket, key string, data []byte) error {
	cfg, err := awsconfig.LoadDefaultConfig(ctx)
	if err != nil {
		return fmt.Errorf("loading AWS config: %w", err)
	}
	s3Client := s3.NewFromConfig(cfg)
	_, err = s3Client.PutObject(ctx, &s3.PutObjectInput{
		Bucket:      aws.String(bucket),
		Key:         aws.String(key),
		Body:        bytes.NewReader(data),
		ContentType: aws.String(inferContentType(key)),
	})
	if err != nil {
		return fmt.Errorf("uploading to s3://%s/%s: %w", bucket, key, err)
	}
	return nil
}

func inferContentType(key string) string {
	if strings.HasSuffix(key, ".html") {
		return "text/html"
	}
	if strings.HasSuffix(key, ".json") {
		return "application/json"
	}
	return "application/octet-stream"
}

// DownloadFromS3 downloads a text file from S3. Returns empty string on error.
func (c *Client) DownloadFromS3(ctx context.Context, bucket, key string) (string, error) {
	cfg, err := awsconfig.LoadDefaultConfig(ctx)
	if err != nil {
		return "", err
	}
	s3Client := s3.NewFromConfig(cfg)
	result, err := s3Client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		return "", err
	}
	defer result.Body.Close()
	data, err := io.ReadAll(result.Body)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// DeleteS3Prefix deletes every object under the given prefix and returns the
// number of objects removed. Used to clear stale training checkpoints before a
// fresh (non-incremental) run so the trainer does not resume from a checkpoint
// produced by a previous run on different data.
func (c *Client) DeleteS3Prefix(ctx context.Context, bucket, prefix string) (int, error) {
	cfg, err := awsconfig.LoadDefaultConfig(ctx)
	if err != nil {
		return 0, fmt.Errorf("loading AWS config: %w", err)
	}
	s3Client := s3.NewFromConfig(cfg)
	paginator := s3.NewListObjectsV2Paginator(s3Client, &s3.ListObjectsV2Input{
		Bucket: aws.String(bucket),
		Prefix: aws.String(prefix),
	})
	deleted := 0
	for paginator.HasMorePages() {
		page, err := paginator.NextPage(ctx)
		if err != nil {
			return deleted, fmt.Errorf("listing s3://%s/%s: %w", bucket, prefix, err)
		}
		for _, obj := range page.Contents {
			if _, err := s3Client.DeleteObject(ctx, &s3.DeleteObjectInput{
				Bucket: aws.String(bucket),
				Key:    obj.Key,
			}); err != nil {
				return deleted, fmt.Errorf("deleting %s: %w", aws.ToString(obj.Key), err)
			}
			deleted++
		}
	}
	return deleted, nil
}

// EnsureBucketEncryption verifies that the S3 bucket has server-side encryption
// enabled. If encryption is not configured, it enables SSE-S3 as a baseline.
// For higher data classifications (Highly Confidential+), customers should
// upgrade to SSE-KMS with a customer-managed CMK.
func (c *Client) EnsureBucketEncryption(ctx context.Context, bucket string) error {
	cfg, err := awsconfig.LoadDefaultConfig(ctx)
	if err != nil {
		return fmt.Errorf("loading AWS config: %w", err)
	}
	s3Client := s3.NewFromConfig(cfg)

	// Check if encryption is already configured
	_, err = s3Client.GetBucketEncryption(ctx, &s3.GetBucketEncryptionInput{
		Bucket: aws.String(bucket),
	})
	if err == nil {
		// Encryption already configured
		return nil
	}

	// Enable SSE-S3 default encryption with bucket key
	_, err = s3Client.PutBucketEncryption(ctx, &s3.PutBucketEncryptionInput{
		Bucket: aws.String(bucket),
		ServerSideEncryptionConfiguration: &s3types.ServerSideEncryptionConfiguration{
			Rules: []s3types.ServerSideEncryptionRule{
				{
					ApplyServerSideEncryptionByDefault: &s3types.ServerSideEncryptionByDefault{
						SSEAlgorithm: s3types.ServerSideEncryptionAes256,
					},
					BucketKeyEnabled: aws.Bool(true),
				},
			},
		},
	})
	if err != nil {
		return fmt.Errorf("enabling encryption on bucket %s: %w", bucket, err)
	}
	return nil
}

// JobStatusInfo holds status information about a K8s Job.
type JobStatusInfo struct {
	Phase    string        // Complete, Failed, Running, Pending
	Reason   string        // failure reason if failed
	Duration time.Duration // time since creation
}

// GetJobStatus returns the current status of a Job.
func (c *Client) GetJobStatus(ctx context.Context, jobName string) (*JobStatusInfo, error) {
	job, err := c.clientset.BatchV1().Jobs(c.namespace).Get(ctx, jobName, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}

	info := &JobStatusInfo{
		Phase: "Running",
	}

	if !job.CreationTimestamp.IsZero() {
		info.Duration = time.Since(job.CreationTimestamp.Time)
	}

	for _, cond := range job.Status.Conditions {
		if cond.Type == batchv1.JobComplete && cond.Status == corev1.ConditionTrue {
			info.Phase = "Complete"
			if job.Status.CompletionTime != nil {
				info.Duration = job.Status.CompletionTime.Sub(job.CreationTimestamp.Time)
			}
			return info, nil
		}
		if cond.Type == batchv1.JobFailed && cond.Status == corev1.ConditionTrue {
			info.Phase = "Failed"
			info.Reason = cond.Reason
			if cond.Message != "" {
				info.Reason = cond.Message
			}
			return info, nil
		}
	}

	if job.Status.Active == 0 && job.Status.Succeeded == 0 && job.Status.Failed == 0 {
		info.Phase = "Pending"
	}

	return info, nil
}

// GetDeploymentReadiness returns the ready and total replica counts for a Deployment.
func (c *Client) GetDeploymentReadiness(ctx context.Context, name string) (int32, int32, error) {
	dep, err := c.clientset.AppsV1().Deployments(c.namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return 0, 0, err
	}
	total := int32(1)
	if dep.Spec.Replicas != nil {
		total = *dep.Spec.Replicas
	}
	return dep.Status.ReadyReplicas, total, nil
}

// CheckS3Object checks if an S3 object exists and returns its size.
func (c *Client) CheckS3Object(ctx context.Context, bucket, key string) (bool, int64, error) {
	cfg, err := awsconfig.LoadDefaultConfig(ctx)
	if err != nil {
		return false, 0, err
	}
	s3Client := s3.NewFromConfig(cfg)
	head, err := s3Client.HeadObject(ctx, &s3.HeadObjectInput{
		Bucket: &bucket,
		Key:    &key,
	})
	if err != nil {
		return false, 0, nil // not found is not an error
	}
	size := int64(0)
	if head.ContentLength != nil {
		size = *head.ContentLength
	}
	return true, size, nil
}

// GetConfigMapData reads a specific key from a ConfigMap.
func (c *Client) GetConfigMapData(ctx context.Context, name, key string) (string, error) {
	cm, err := c.clientset.CoreV1().ConfigMaps(c.namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	data, ok := cm.Data[key]
	if !ok {
		return "", fmt.Errorf("key %q not found in ConfigMap %s", key, name)
	}
	return data, nil
}

// RunEphemeralPod creates a short-lived pod, waits for completion, and returns its stdout.
// Used for in-cluster operations like fetching logs from internal services.
func (c *Client) RunEphemeralPod(ctx context.Context, name, image string, command []string) (string, error) {
	noEscalation := false
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: c.namespace,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by": "slemify",
				"slemify.io/ephemeral":         "true",
			},
		},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyNever,
			NodeSelector: map[string]string{
				"slemify.io/workload": "slm",
			},
			Tolerations: []corev1.Toleration{
				{
					Key:      "slemify.io/slm",
					Operator: corev1.TolerationOpExists,
					Effect:   corev1.TaintEffectNoSchedule,
				},
			},
			Containers: []corev1.Container{
				{
					Name:    "run",
					Image:   image,
					Command: command,
					SecurityContext: &corev1.SecurityContext{
						AllowPrivilegeEscalation: &noEscalation,
					},
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("100m"),
							corev1.ResourceMemory: resource.MustParse("128Mi"),
						},
						Limits: corev1.ResourceList{
							corev1.ResourceMemory: resource.MustParse("128Mi"),
						},
					},
				},
			},
		},
	}

	// Delete any existing pod with the same name
	c.clientset.CoreV1().Pods(c.namespace).Delete(ctx, name, metav1.DeleteOptions{})
	time.Sleep(2 * time.Second)

	_, err := c.clientset.CoreV1().Pods(c.namespace).Create(ctx, pod, metav1.CreateOptions{})
	if err != nil {
		return "", fmt.Errorf("creating ephemeral pod: %w", err)
	}
	defer c.clientset.CoreV1().Pods(c.namespace).Delete(ctx, name, metav1.DeleteOptions{})

	// Wait for pod to complete
	for i := 0; i < 60; i++ {
		p, err := c.clientset.CoreV1().Pods(c.namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return "", err
		}
		if p.Status.Phase == corev1.PodSucceeded {
			break
		}
		if p.Status.Phase == corev1.PodFailed {
			return "", fmt.Errorf("ephemeral pod failed")
		}
		time.Sleep(2 * time.Second)
	}

	// Get logs
	req := c.clientset.CoreV1().Pods(c.namespace).GetLogs(name, &corev1.PodLogOptions{})
	stream, err := req.Stream(ctx)
	if err != nil {
		return "", fmt.Errorf("getting pod logs: %w", err)
	}
	defer stream.Close()

	logData, err := io.ReadAll(stream)
	if err != nil {
		return "", fmt.Errorf("reading pod logs: %w", err)
	}
	return string(logData), nil
}

// IsNodeOverlayEnabled checks if the Karpenter NodeOverlay feature gate is enabled
// by verifying the nodeoverlays.karpenter.sh CRD exists in the cluster.
func (c *Client) IsNodeOverlayEnabled(ctx context.Context) bool {
	gvr := schema.GroupVersionResource{
		Group:    "karpenter.sh",
		Version:  "v1alpha1",
		Resource: "nodeoverlays",
	}
	_, err := c.dynamicClient.Resource(gvr).List(ctx, metav1.ListOptions{Limit: 1})
	return err == nil
}

// IsMountpointCSIEnabled checks if the Mountpoint for Amazon S3 CSI driver
// is installed by looking for the s3.csi.aws.com CSIDriver resource.
func (c *Client) IsMountpointCSIEnabled(ctx context.Context) bool {
	_, err := c.clientset.StorageV1().CSIDrivers().Get(ctx, "s3.csi.aws.com", metav1.GetOptions{})
	return err == nil
}

// GetFailedPodCount returns the number of failed pods for a Job and the last error message.
func (c *Client) GetFailedPodCount(ctx context.Context, jobName string) (int, string) {
	pods, err := c.clientset.CoreV1().Pods(c.namespace).List(ctx, metav1.ListOptions{
		LabelSelector: "job-name=" + jobName,
	})
	if err != nil {
		return 0, ""
	}
	failed := 0
	lastReason := ""
	for _, pod := range pods.Items {
		if pod.Status.Phase == corev1.PodFailed {
			failed++
			// Check pod-level reason first (e.g., UnexpectedAdmissionError)
			if pod.Status.Reason != "" {
				lastReason = pod.Status.Reason
			}
			// Fall back to container-level reason
			if lastReason == "" {
				for _, cs := range pod.Status.ContainerStatuses {
					if cs.State.Terminated != nil && cs.State.Terminated.Reason != "" {
						lastReason = cs.State.Terminated.Reason
					}
				}
			}
			if lastReason == "" {
				lastReason = string(pod.Status.Phase)
			}
		}
	}
	return failed, lastReason
}
