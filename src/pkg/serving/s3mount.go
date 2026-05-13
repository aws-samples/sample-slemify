// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package serving

import (
	"fmt"
)

// S3MountManifests generates PersistentVolume and PersistentVolumeClaim YAML
// for mounting an S3 bucket via the Mountpoint for Amazon S3 CSI driver.
// The model file is accessed read-only via mmap, eliminating the need for
// an init container to download the GGUF file from S3.
//
// This requires the Mountpoint for Amazon S3 CSI driver to be installed
// on the cluster (EKS add-on: aws-mountpoint-s3-csi-driver).
func S3MountManifests(projectName, bucket, ns string) string {
	pvName := fmt.Sprintf("%s-model-s3", projectName)
	pvcName := fmt.Sprintf("%s-model-s3", projectName)
	// Mount only the models/<project>/ prefix so the pod sees the GGUF file at root
	prefix := fmt.Sprintf("models/%s/", projectName)

	return fmt.Sprintf(`apiVersion: v1
kind: PersistentVolume
metadata:
  name: %s
  labels:
    app.kubernetes.io/managed-by: slemify
    slemify.io/project: %s
spec:
  accessModes:
    - ReadOnlyMany
  capacity:
    storage: 100Gi
  storageClassName: ""
  claimRef:
    namespace: %s
    name: %s
  mountOptions:
    - read-only
    - allow-other
    - prefix %s
  csi:
    driver: s3.csi.aws.com
    volumeHandle: %s
    volumeAttributes:
      bucketName: %s
      authenticationSource: pod
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: %s
  namespace: %s
  labels:
    app.kubernetes.io/managed-by: slemify
    slemify.io/project: %s
spec:
  accessModes:
    - ReadOnlyMany
  storageClassName: ""
  resources:
    requests:
      storage: 100Gi
  volumeName: %s
`, pvName, projectName, ns, pvcName, prefix, pvName, bucket, pvcName, ns, projectName, pvName)
}

// S3ModelVolumeName returns the PVC name for the S3-mounted model volume.
func S3ModelVolumeName(projectName string) string {
	return fmt.Sprintf("%s-model-s3", projectName)
}
