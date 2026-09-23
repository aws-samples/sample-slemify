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
  buildspec.yaml         start image builds, terraform apply (VPC, bucket, cluster, everything but the
                         S3 CSI addon), start the GGUF conversion early, join manifests, seed, signal
                         CloudFormation READY, then finish the addon and wait for the conversion
  buildspec-images.yaml  build the six images for one architecture and push to the account's ECR
  buildspec-gguf.yaml    convert the analyst base model to GGUF on the 2XLARGE fleet, upload to the bucket
  terraform/             VPC, EKS Auto Mode cluster, addons, model bucket, IAM
  seed/
    seed.sh              NodePools, pre-warm, OpenSearch, Titan index, example data, triage data stage,
                         retriever (all stages), tuned index, demo in monolith mode, smoke query
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

Between the Titan index and the demo it puts the Slemify projects where the
module pages start: uploads `data/` to the model bucket, runs the triage
project's data stage only (`slemify deploy --until data`, so attendees run
training in module 1), deploys the retriever end to end, and builds the tuned
knowledge index against the served encoder (`make recall` in module 2 scores
both indexes). This part needs the `slemify` binary; `buildspec.yaml`
installs the release named by `SLEMIFY_VERSION`, and the seed skips it when
the binary is missing. Slemify reads `SLEMIFY_BUCKET` and
`SLEMIFY_BEDROCK_MODEL` from the environment in place of the bucket and model
in the shipped `expert.yaml` files; the IDE sets the same two variables.

Images: the seed expects `slemify/k8s-autoscaling-orchestrator` and
`slemify/k8s-autoscaling-reranker` in the account's ECR unless `DEMO_IMAGE`
and `RERANKER_IMAGE` point elsewhere. The images are built in the account
because vended accounts have no registry of their own and a shared one would
need a cross-account grant per account.

GGUF: the analyst's base model (a 30B mixture-of-experts) needs about 140 GB
of scratch disk to convert, and Auto Mode's default NodeClass gives nodes an
80 GiB volume, so the in-cluster convert Job is evicted every time. The
conversion runs in CodeBuild instead (`buildspec-gguf.yaml`, 14 minutes on
`BUILD_GENERAL1_2XLARGE`) and `slemify deploy` skips the convert when the
file is already in the bucket.

## Two-phase readiness

Provisioning signals CloudFormation (and so Workshop Studio) that the event is
ready as soon as the seed's smoke query passes: at that point everything
modules 0 to 2 use is up. Two assets that only module 3 needs are finished
afterwards, still inside the same build:

- the Mountpoint S3 CSI addon, which takes 8 to 9 minutes to report ACTIVE
  because its pods cannot schedule until the first node exists. It is applied
  with `terraform apply -target=aws_eks_addon.s3_csi` (retried), so it stays
  in Terraform state and a re-apply converges it;
- the analyst GGUF conversion, started as soon as the amd64 `gguf-convert`
  image is pushed (it needs only the bucket and that image, not the cluster),
  so it normally finishes before the seed does.

Measured: the event signals ready at build minute 25 to 26 (about 27
minutes of wall clock including the account vend), against 42 before, and
both assets are ACTIVE and in the bucket a few minutes later, well before
anyone reaches module 3.

If either fails, the build shows FAILED in CodeBuild while the event stays
usable. Module 3's page has attendees run `make check-infra`, which reports
each asset as ok, wait, or missing, and `make check-infra REPAIR=1` re-runs
this provisioning build. The re-run is idempotent: Terraform converges, the
conversion skips when the file exists, the seed skips what is present, and
the CloudFormation signal is skipped because the stack is already complete.
The IDE role has `codebuild:StartBuild` on the two projects for this. Without
the addon, `slemify deploy` falls back to downloading the model into the pod
(slower start, same answers); without the GGUF, the analyst cannot deploy
until the repair finishes.

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
