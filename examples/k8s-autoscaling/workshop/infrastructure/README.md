# Workshop provisioning

What CodeBuild runs to put an AWS account into the workshop's starting state.
Workshop Studio deploys one CloudFormation stack per team from the workshop
content repository; that stack creates the attendee IDE and a CodeBuild
project whose source is this repository at a release tag. CodeBuild runs
`buildspec.yaml`, which applies the Terraform and then the seed. The
CloudFormation template, `contentspec.yaml`, and the participant IAM policy
live in the content repository, not here.

```
infrastructure/
  buildspec.yaml         terraform apply (VPC, cluster, rest), then seed, then signal CloudFormation
  terraform/             VPC, EKS Auto Mode cluster, addons, model bucket, IAM
  seed/
    seed.sh              NodePools, pre-warm, OpenSearch, Titan index, demo in monolith mode, smoke query
    smoke.sh             One query; asserts the cost and total events
    slm-nodepools.yaml   The three weighted Slemify SLM NodePools (Auto Mode)
    prewarm.yaml         Pause pod that brings one SLM node online early
    storageclass.yaml    gp3 StorageClass (Auto Mode ships none)
```

## Terraform

Creates the cluster VPC (three AZs, one NAT), an EKS Auto Mode cluster
(`system` and `general-purpose` pools, access-entry authentication),
cluster-admin access entries for the participant role and the IDE instance
role, the Pod Identity agent and Mountpoint S3 CSI addons, a KMS-encrypted
model bucket, and the orchestrator's Bedrock pod-identity role.

Inputs come from CodeBuild environment variables set by the stack
(`EKS_CLUSTER_NAME`, `EKS_CLUSTER_VERSION`, `BEDROCK_REGION`,
`PARTICIPANT_ROLE_ARN`, `IDE_ROLE_ARN`, `TF_STATE_BUCKET`, `STACK_NAME`).
`buildspec.yaml` writes the S3 backend to `backend.tf` from those.

Slemify creates its own per-project IAM role, service account, and pod
identity association when an attendee runs `slemify deploy`
(`pkg/k8s/identity.go`); those are not in Terraform.

## Seed

`seed/seed.sh` is idempotent. It applies the three Slemify SLM NodePools (a
static copy of what `pkg/serving/nodepool.go` renders for Auto Mode) and a
pre-warm pod so one SLM node is online before module 1, installs single-node
OpenSearch, builds the Titan knowledge index the monolith reads, deploys the
demo in monolith mode (`TRIAGE=off EMBED=bedrock RERANK=off ANALYST=llm
GATE=off`) with the Bedrock region and model set, and runs one smoke query so
the first attendee query is not the first query.

Images: the seed expects `slemify/k8s-autoscaling-orchestrator` and
`slemify/k8s-autoscaling-reranker` in the account's ECR unless `DEMO_IMAGE`
and `RERANKER_IMAGE` point elsewhere.

## Running it by hand

Useful while iterating on the Terraform or the seed. Same commands CodeBuild
runs, from a machine with `terraform`, `kubectl`, `helm`, `aws`, `python3`,
and `jq`:

```bash
cd examples/k8s-autoscaling/workshop/infrastructure
export AWS_REGION=us-west-2 EKS_CLUSTER_NAME=slemify-workshop BEDROCK_REGION=us-west-2
export TF_VAR_region=$AWS_REGION TF_VAR_cluster_name=$EKS_CLUSTER_NAME TF_VAR_bedrock_region=$BEDROCK_REGION

terraform -chdir=terraform init
terraform -chdir=terraform apply -target=module.vpc
terraform -chdir=terraform apply -target=module.eks
terraform -chdir=terraform apply
bash seed/seed.sh

# Tear down: remove the namespace first so load balancers and volumes release.
kubectl delete namespace slemify --timeout=300s
terraform -chdir=terraform destroy
```
