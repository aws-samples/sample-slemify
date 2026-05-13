// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"

	"github.com/aws-samples/sample-slemify/pkg/build"
	"github.com/aws-samples/sample-slemify/pkg/config"
	"github.com/aws-samples/sample-slemify/pkg/data"
	"github.com/aws-samples/sample-slemify/pkg/k8s"
	"github.com/aws-samples/sample-slemify/pkg/pipeline"
	"github.com/aws-samples/sample-slemify/pkg/serving"
	"github.com/aws-samples/sample-slemify/pkg/training"
	"github.com/spf13/cobra"
)

var deployCmd = &cobra.Command{
	Use:   "deploy",
	Short: "Run the full pipeline: data → training → quantize → serving",
	RunE: func(cmd *cobra.Command, args []string) error {
		stage, _ := cmd.Flags().GetString("stage")
		dryRun, _ := cmd.Flags().GetBool("dry-run")
		ctx := cmd.Context()

		cfg, warnings, err := config.Load(cfgFile)
		if err != nil {
			return fmt.Errorf("failed to load config: %w", err)
		}
		for _, w := range warnings {
			fmt.Printf("⚠ Warning: %s\n", w)
		}
		if errs := config.Validate(cfg); len(errs) > 0 {
			for _, e := range errs {
				fmt.Printf("  ✗ %s: %s\n", e.Field, e.Message)
			}
			return fmt.Errorf("config has %d validation error(s)", len(errs))
		}

		return deploySingleExpert(ctx, cmd, cfg, stage, dryRun)
	},
}

func deploySingleExpert(ctx context.Context, cmd *cobra.Command, cfg *config.ExpertConfig, stage string, dryRun bool) error {
	sized := config.AutoSize(cfg.Model, cfg.Data, cfg.Training)

	// Load output token stats from S3 (written by data pipeline)
	// These drive max_tokens and reasoning_budget for inference
	if !dryRun {
		if stats, err := loadOutputStats(ctx, cfg); err == nil {
			sized.MaxOutputTokens = stats.P95 + stats.P95/5 // p95 + 20% headroom
			if cfg.Project.IsFreeForm() {
				sized.ReasoningBudget = stats.Avg / 2
			}
		}
	}

	// Load persisted state from S3 (enables resume after crashes)
	var state *pipeline.State
	store, storeErr := pipeline.NewStateStore(ctx, cfg.Data.Bucket)
	if storeErr == nil {
		state, _ = store.Load(ctx, cfg.Project.Name)
	} else {
		state = pipeline.NewState(cfg.Project.Name)
	}

	runner := pipeline.NewRunner(cfg.Project.Name, state)
	if storeErr == nil {
		runner.SetStateStore(store)
	}

	pc := pipeline.NewPipelineContext()

	// Set image registry for all stages
	// Auto-detect ECR registry from AWS account when not explicitly set
	if imageRegistry == "" && !dryRun {
		ecrMgr, err := build.NewECRManager(ctx)
		if err == nil {
			imageRegistry = ecrMgr.RegistryURL() + "/slemify"
			fmt.Printf("Auto-detected ECR registry: %s\n", imageRegistry)
		} else {
			return fmt.Errorf("no --image-registry specified and ECR auto-detection failed: %w", err)
		}
	}
	if imageRegistry != "" {
		// ECR repos are under slemify/ prefix (e.g., 123456.dkr.ecr.../slemify/data-pipeline)
		if !strings.HasSuffix(imageRegistry, "/slemify") && strings.Contains(imageRegistry, ".ecr.") {
			imageRegistry = imageRegistry + "/slemify"
		}
		pc.Registry = imageRegistry
	}

	var client *k8s.Client
	var k8sErr error
	if !dryRun {
		client, k8sErr = k8s.NewClient(kubeconfig, namespace)
	} else {
		k8sErr = fmt.Errorf("dry-run mode")
	}

	if k8sErr == nil {
		if err := setupClusterInfrastructure(ctx, client, cfg, sized, pc); err != nil {
			return err
		}
		runner.RegisterStage(pipeline.StageData, data.Stage(client, cfg, namespace, pc))
		runner.RegisterStage(pipeline.StageTraining, training.Stage(client, cfg, sized, namespace, pc))
		runner.RegisterStage(pipeline.StageQuantize, func(ctx context.Context) ([]string, error) { return nil, nil })
		runner.RegisterStage(pipeline.StageServing, serving.Stage(client, cfg, sized, namespace, pc))
	} else {
		fmt.Printf("⚠ No K8s cluster available (%v), running in dry-run mode\n\n", k8sErr)
		registerDryRunStages(runner, cfg, sized, pc)
	}

	fmt.Printf("Deploying %s (%s)\n", cfg.Project.Name, cfg.Project.Domain)
	fmt.Printf("Model: %s → %s (training) → %s (inference)\n",
		cfg.Model.Base, sized.TrainingInstance, sized.InferenceInstance)
	fmt.Printf("Namespace: %s\n\n", namespace)

	var startStage pipeline.Stage
	if stage != "" {
		var err error
		startStage, err = pipeline.ParseStage(stage)
		if err != nil {
			return err
		}
	}

	noWait, _ := cmd.Flags().GetBool("no-wait")
	if noWait {
		if startStage == "" {
			return fmt.Errorf("--no-wait requires --stage to specify which stage to submit")
		}
		pc.NoWait = true
		runner.SetNoWait(true)
	}

	if err := runner.Run(ctx, startStage); err != nil {
		return err
	}

	if noWait {
		fmt.Printf("\n📤 Stage %s submitted. Check progress with:\n", startStage)
		fmt.Printf("   slemify status %s\n", cfg.Project.Name)
		return nil
	}

	fmt.Printf("\n✅ %s deployed successfully\n", cfg.Project.Name)
	fmt.Printf("   Inference:     %s-inference.%s.svc.cluster.local:8080\n", cfg.Project.Name, namespace)
	fmt.Printf("   Model:         %s → %s (%s GGUF)\n", cfg.Model.Base, sized.InferenceInstance, cfg.Model.QuantizeLabel())
	// Spot pricing estimate for inference
	spotPricing := map[string]float64{
		"c8g.medium": 25, "c8g.xlarge": 50, "c8g.2xlarge": 80,
		"c8g.4xlarge": 140, "c8g.8xlarge": 280, "r8g.8xlarge": 320,
	}
	if price, ok := spotPricing[sized.InferenceInstance]; ok {
		fmt.Printf("   Est. monthly:  ~$%.0f/mo (Spot)\n", price)
	}
	fmt.Printf("\n   Next: slemify analyze --config %s\n", cfgFile)
	return nil
}

// setupClusterInfrastructure creates the namespace, Pod Identity, Karpenter NodePools,
// NodeOverlays, and detects CSI driver availability.
func setupClusterInfrastructure(ctx context.Context, client *k8s.Client, cfg *config.ExpertConfig, sized config.SizedConfig, pc *pipeline.PipelineContext) error {
	if err := client.EnsureNamespace(ctx); err != nil {
		return fmt.Errorf("creating namespace %s: %w", namespace, err)
	}

	fmt.Println("Verifying S3 bucket encryption...")
	if err := client.EnsureBucketEncryption(ctx, cfg.Data.Bucket); err != nil {
		fmt.Printf("  ⚠ Could not verify bucket encryption: %v\n", err)
		fmt.Println("  Ensure server-side encryption is enabled on your bucket.")
	} else {
		fmt.Println("  S3 bucket encryption verified")
	}
	fmt.Println()

	fmt.Println("Setting up Pod Identity...")
	clusterName, err := k8s.DetectClusterName(kubeconfig)
	if err != nil {
		return fmt.Errorf("detecting cluster name: %w", err)
	}
	nodeRole := client.DetectNodeRole(ctx, clusterName)
	podID, err := client.EnsurePodIdentity(ctx, clusterName, cfg.Data.Bucket, cfg.Project.Name)
	if err != nil {
		return fmt.Errorf("setting up Pod Identity: %w", err)
	}
	pc.ServiceAccount = podID.ServiceAccountName
	fmt.Println()

	fmt.Println("Setting up Karpenter NodePools...")
	slmNodeClass := serving.SLMEC2NodeClassManifest(clusterName, nodeRole, cfg.Project.Name)
	slmNodePool := serving.SLMNodePoolManifest(sized)
	if err := client.ApplyYAML(ctx, []byte(slmNodeClass)); err != nil {
		return fmt.Errorf("applying slemify-slm EC2NodeClass: %w", err)
	}
	if err := client.ApplyYAML(ctx, []byte(slmNodePool)); err != nil {
		return fmt.Errorf("applying slemify-slm NodePool: %w", err)
	}
	gpuNodeClass := training.GPUEC2NodeClassManifest(clusterName, nodeRole, cfg.Project.Name)
	gpuNodePool := training.GPUNodePoolManifest(cfg, sized)
	if err := client.ApplyYAML(ctx, []byte(gpuNodeClass)); err != nil {
		return fmt.Errorf("applying slemify-gpu EC2NodeClass: %w", err)
	}
	if err := client.ApplyYAML(ctx, []byte(gpuNodePool)); err != nil {
		return fmt.Errorf("applying slemify-gpu NodePool: %w", err)
	}

	fmt.Println("Setting up NodeOverlays...")
	if client.IsNodeOverlayEnabled(ctx) {
		nodeOverlays := serving.NodeOverlayManifests()
		for _, doc := range pipeline.SplitYAMLDocs(nodeOverlays) {
			if err := client.ApplyYAML(ctx, []byte(doc)); err != nil {
				fmt.Printf("  ⚠ Failed to apply NodeOverlay: %v\n", err)
				break
			}
		}
		fmt.Println("  NodeOverlays applied (preferring latest generation instances)")
	} else {
		fmt.Println("  ⚠ NodeOverlay feature gate not enabled in Karpenter")
		fmt.Println("  Enable it with: helm upgrade karpenter oci://public.ecr.aws/karpenter/karpenter --set \"settings.featureGates.nodeOverlay=true\" --reuse-values -n karpenter")
		fmt.Println("  Without NodeOverlays, Karpenter selects instances by lowest price (any generation)")
	}
	fmt.Println()

	fmt.Println("Checking Mountpoint for S3 CSI driver...")
	if client.IsMountpointCSIEnabled(ctx) {
		pc.UseS3Mount = true
		fmt.Println("  Mountpoint CSI driver detected — models will be mounted directly from S3")
		fmt.Println("  (no init container download, llama.cpp reads via mmap)")
	} else {
		pc.UseS3Mount = false
		fmt.Println("  Mountpoint CSI driver not found — models will be downloaded via init container")
		fmt.Println("  Install it for faster pod startup: https://docs.aws.amazon.com/eks/latest/userguide/s3-csi-create.html")
	}
	fmt.Println()
	return nil
}

func registerDryRunStages(runner *pipeline.Runner, cfg *config.ExpertConfig, sized config.SizedConfig, pc *pipeline.PipelineContext) {
	runner.RegisterStage(pipeline.StageData, func(ctx context.Context) ([]string, error) {
		fmt.Printf("  Bucket: s3://%s/%s\n", cfg.Data.Bucket, cfg.Data.Path)
		fmt.Printf("  Sources: %d\n", len(cfg.Data.Sources))
		fmt.Printf("  Synthetic: %d pairs via %s\n", cfg.Data.Synthetic.Pairs, cfg.Data.Synthetic.Model)
		return []string{
			fmt.Sprintf("s3://%s/%s/processed/train.jsonl", cfg.Data.Bucket, cfg.Project.Name),
			fmt.Sprintf("s3://%s/%s/processed/eval.jsonl", cfg.Data.Bucket, cfg.Project.Name),
		}, nil
	})
	runner.RegisterStage(pipeline.StageTraining, func(ctx context.Context) ([]string, error) {
		fmt.Printf("  Base model: %s\n", cfg.Model.Base)
		fmt.Printf("  Instance: %s (%s)\n", sized.TrainingInstance, sized.TrainingGPU)
		fmt.Printf("  Spot: %v\n", cfg.Training.Spot)
		fmt.Printf("  Epochs: %d, LR: %g\n", sized.Epochs, sized.LearningRate)
		return []string{
			fmt.Sprintf("s3://%s/models/%s/full/", cfg.Data.Bucket, cfg.Project.Name),
		}, nil
	})
	runner.RegisterStage(pipeline.StageQuantize, func(ctx context.Context) ([]string, error) {
		fmt.Printf("  Quantization: %s (GGUF)\n", cfg.Model.QuantizeLabel())
		return []string{
			fmt.Sprintf("s3://%s/models/%s/%s", cfg.Data.Bucket, cfg.Project.Name, cfg.Model.GGUFFilename()),
		}, nil
	})
	runner.RegisterStage(pipeline.StageServing, func(ctx context.Context) ([]string, error) {
		m := serving.GenerateInferenceManifests(cfg, sized, namespace, pc)
		fmt.Printf("  Instance: %s\n", sized.InferenceInstance)
		fmt.Printf("  Deployment: %s\n", m.Deployment.Name)
		fmt.Printf("  Service: %s (ClusterIP:8080)\n", m.Service.Name)
		fmt.Printf("  PDB: minAvailable=1\n")
		fmt.Printf("  Metrics: /metrics (Prometheus-compatible)\n")
		return []string{
			fmt.Sprintf("%s-inference.%s.svc.cluster.local:8080", cfg.Project.Name, namespace),
		}, nil
	})
}

func init() {
	deployCmd.Flags().String("stage", "", "Start from a specific stage (data, training, quantize, serving)")
	deployCmd.Flags().Bool("dry-run", false, "Show what would be deployed without connecting to a cluster")
	deployCmd.Flags().Bool("no-wait", false, "Submit the stage and exit without waiting for completion (use with --stage)")
	rootCmd.AddCommand(deployCmd)
}

// outputStats holds token statistics from the data pipeline.
type outputStats struct {
	Max int `json:"max_output_tokens"`
	Avg int `json:"avg_output_tokens"`
	P95 int `json:"p95_output_tokens"`
}

// loadOutputStats reads output_stats.json from S3 (written by data pipeline).
func loadOutputStats(ctx context.Context, cfg *config.ExpertConfig) (*outputStats, error) {
	key := fmt.Sprintf("%s/processed/output_stats.json", cfg.Project.Name)

	awsCfg, err := awsconfig.LoadDefaultConfig(ctx)
	if err != nil {
		return nil, err
	}
	s3Client := s3.NewFromConfig(awsCfg)
	result, err := s3Client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: &cfg.Data.Bucket,
		Key:    &key,
	})
	if err != nil {
		return nil, err
	}
	defer result.Body.Close()

	var stats outputStats
	if err := json.NewDecoder(result.Body).Decode(&stats); err != nil {
		return nil, err
	}
	fmt.Printf("Output stats from data: max=%d, avg=%d, p95=%d tokens\n", stats.Max, stats.Avg, stats.P95)
	return &stats, nil
}
