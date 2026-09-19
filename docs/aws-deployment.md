# AWS deployment

## Why ECS Fargate

The BRD offers SageMaker, ECS/Fargate or Lambda. This deploys to **ECS
Fargate**, and the reason is the shape of the application rather than a
preference:

- **Lambda cannot host the UI.** Streamlit is a long-lived, stateful web
  server holding a websocket per session. Lambda's request/response model
  does not fit it, so a Lambda deployment would still need somewhere else for
  the front end.
- **A SageMaker endpoint solves half the problem.** It serves the model well,
  but the web application is the actual deliverable and would need separate
  hosting anyway — two systems where one suffices at this scale.
- **Fargate runs both** from a single image, with no servers to patch. The
  API and UI are two services differing only in their `command`.

If the project later grows real inference-scaling needs — autoscaling on
model latency, multi-model endpoints, batch transform — SageMaker becomes the
better home for the model, with the UI staying on Fargate.

## Components

| Piece | Purpose |
| --- | --- |
| ECR repository | Holds the single image both services run |
| S3 bucket | Dataset (`FR-1`) and trained checkpoints |
| ECS cluster + 2 services | `leaf-api` on :8000, `leaf-ui` on :8501 |
| CloudWatch log groups | `/ecs/leaf-api`, `/ecs/leaf-ui` |
| ALB (optional) | Public entry point, TLS termination |
| IAM roles | Deploy role, task execution role, task role |

The image is built once and tagged with the commit SHA, so a running task is
always traceable to the source that produced it.

## One-time setup

Replace `ACCOUNT_ID`, `AWS_REGION` and `BUCKET_NAME` throughout — including
in `infra/ecs-task-definition*.json`.

### 1. ECR and S3

```bash
aws ecr create-repository --repository-name medicinal-leaf \
  --image-scanning-configuration scanOnPush=true

aws s3api create-bucket --bucket BUCKET_NAME \
  --create-bucket-configuration LocationConstraint=AWS_REGION
aws s3api put-public-access-block --bucket BUCKET_NAME \
  --public-access-block-configuration \
  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

Upload the dataset and a trained checkpoint:

```bash
aws s3 sync ./Data s3://BUCKET_NAME/datasets/medicinal-leaf/v1
aws s3 cp artifacts/checkpoints/best.pt s3://BUCKET_NAME/models/best.pt
aws s3 cp artifacts/checkpoints/best.meta.json s3://BUCKET_NAME/models/best.meta.json
```

### 2. GitHub OIDC provider

This is what lets the pipeline deploy **without storing AWS keys in GitHub**
(NFR-4). Create it once per account:

```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
```

### 3. Deploy role (assumed by GitHub Actions)

Trust policy — note the `sub` is the **environment** form, because the deploy
job declares `environment: production`. It would be `...:ref:refs/heads/main`
without that:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
          "token.actions.githubusercontent.com:sub": "repo:nileshkkolekar/Classifying-Indian-Medicinal-Leaf:environment:production"
        }
      }
    }
  ]
}
```

Permissions — scoped to this repository and these services (NFR-7):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "EcrAuthIsAccountWide",
      "Effect": "Allow",
      "Action": "ecr:GetAuthorizationToken",
      "Resource": "*"
    },
    {
      "Sid": "PushOnlyToThisRepository",
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:CompleteLayerUpload",
        "ecr:InitiateLayerUpload",
        "ecr:PutImage",
        "ecr:UploadLayerPart",
        "ecr:BatchGetImage"
      ],
      "Resource": "arn:aws:ecr:AWS_REGION:ACCOUNT_ID:repository/medicinal-leaf"
    },
    {
      "Sid": "RegisterTaskDefinitions",
      "Effect": "Allow",
      "Action": ["ecs:RegisterTaskDefinition", "ecs:DescribeTaskDefinition"],
      "Resource": "*"
    },
    {
      "Sid": "UpdateOnlyTheseServices",
      "Effect": "Allow",
      "Action": ["ecs:UpdateService", "ecs:DescribeServices"],
      "Resource": [
        "arn:aws:ecs:AWS_REGION:ACCOUNT_ID:service/leaf-cluster/leaf-api",
        "arn:aws:ecs:AWS_REGION:ACCOUNT_ID:service/leaf-cluster/leaf-ui"
      ]
    },
    {
      "Sid": "PassOnlyTheTaskRoles",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": [
        "arn:aws:iam::ACCOUNT_ID:role/leafEcsExecutionRole",
        "arn:aws:iam::ACCOUNT_ID:role/leafEcsTaskRole"
      ],
      "Condition": { "StringEquals": { "iam:PassedToService": "ecs-tasks.amazonaws.com" } }
    }
  ]
}
```

`iam:PassRole` is the one to get right: unscoped, it lets anyone who can
register a task definition run a container as *any* role in the account.

### 4. Task execution role

Trusts `ecs-tasks.amazonaws.com`; attach the AWS-managed
`AmazonECSTaskExecutionRolePolicy` (pull from ECR, write to CloudWatch). This
role belongs to the ECS agent, not to the application.

### 5. Task role

Also trusts `ecs-tasks.amazonaws.com`. This is what the *application* gets,
so it should reach exactly the two prefixes it reads and nothing else:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadModelAndDataset",
      "Effect": "Allow",
      "Action": "s3:GetObject",
      "Resource": [
        "arn:aws:s3:::BUCKET_NAME/models/*",
        "arn:aws:s3:::BUCKET_NAME/datasets/*"
      ]
    },
    {
      "Sid": "ListOnlyThosePrefixes",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::BUCKET_NAME",
      "Condition": { "StringLike": { "s3:prefix": ["models/*", "datasets/*"] } }
    }
  ]
}
```

Read-only on purpose: the serving container never needs to write to S3.

### 6. Cluster and services

```bash
aws ecs create-cluster --cluster-name leaf-cluster
aws ecs register-task-definition --cli-input-json file://infra/ecs-task-definition.json
aws ecs create-service --cluster leaf-cluster --service-name leaf-api \
  --task-definition leaf-api --desired-count 1 --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[SUBNET_ID],securityGroups=[SG_ID],assignPublicIp=ENABLED}"
```

Repeat for `leaf-ui`. The UI reaches the API by service name, so enable ECS
Service Connect or Cloud Map on the cluster and set
`MLC_SERVING__API_BASE_URL` to the resulting internal DNS name.

## GitHub configuration

**Secrets** (Settings → Secrets and variables → Actions):

| Name | Value |
| --- | --- |
| `AWS_DEPLOY_ROLE_ARN` | `arn:aws:iam::ACCOUNT_ID:role/leafGitHubDeployRole` |

**Variables** — CD skips cleanly if these are unset, so an unconfigured fork
does not fail on every push:

| Name | Example |
| --- | --- |
| `AWS_REGION` | `ap-south-1` |
| `ECR_REPOSITORY` | `medicinal-leaf` |
| `ECS_CLUSTER` | `leaf-cluster` |
| `ECS_API_SERVICE` | `leaf-api` |
| `ECS_UI_SERVICE` | `leaf-ui` |

No AWS access key or secret is stored anywhere — the role is assumed through
OIDC for the duration of a job.

## The pipeline

CI runs on every push and pull request: lint, format, type-check, tests with
a 70% coverage gate, and a Docker build. CD triggers on CI **succeeding** on
`main` (`workflow_run`), so a merge that breaks the build never reaches AWS.
It then assumes the role, pushes the image tagged with the commit SHA, and
updates both services, waiting for steady state so a failed rollout fails the
workflow rather than passing silently.

## Checkpoint loading

The image contains no model. On startup the API reads
`MLC_AWS__CHECKPOINT_URI` and downloads the checkpoint to
`/app/artifacts/checkpoints/`. A missing or unreachable checkpoint is logged
and `/health` reports `degraded` — the task stays up and explains itself
rather than crash-looping.

This is also why `readonlyRootFilesystem` is `false`. To tighten it, attach a
writable volume at `/app/artifacts` and set the flag to `true`.

## Cost notes

Fargate bills per vCPU-second and GB-second while tasks run. The API is sized
1 vCPU / 3 GB for CPU inference; the UI 0.5 vCPU / 1 GB. Scale both to zero
when idle during development:

```bash
aws ecs update-service --cluster leaf-cluster --service leaf-api --desired-count 0
```

Set an AWS Budget with an alert before the first long-running deployment.

## Before this is production

- Put both services behind an ALB with TLS; do not expose task IPs directly.
- Add authentication — the API currently accepts uploads from anyone who can
  reach it.
- Add request rate limiting; the upload limits bound a single request's cost,
  not the number of requests.
- Enable ECR image scanning findings review and pin the base image by digest.
- Set CloudWatch alarms on 5xx rate and on the `unable_to_classify` share,
  which is an early signal of drift or of a changed capture environment.
