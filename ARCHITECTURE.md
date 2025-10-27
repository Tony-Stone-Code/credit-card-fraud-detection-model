# System Architecture Diagrams

## High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Client Layer                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Web Portal   │  │ Mobile App   │  │ Payment      │              │
│  │              │  │              │  │ Gateway      │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
└─────────┼──────────────────┼──────────────────┼──────────────────────┘
          │                  │                  │
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     API Gateway / Load Balancer                      │
│                         (nginx / AWS ALB)                            │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ API Instance 1   │  │ API Instance 2   │  │ API Instance N   │
│  FastAPI Server  │  │  FastAPI Server  │  │  FastAPI Server  │
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
         │                     │                      │
         └─────────────────────┼──────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
    ┌────────────┐      ┌────────────┐     ┌────────────┐
    │ Model      │      │ Redis      │     │ Prometheus │
    │ Storage    │      │ Cache      │     │ Monitoring │
    │ (S3/GCS)   │      │            │     │            │
    └────────────┘      └────────────┘     └────────────┘
           │
           ▼
    ┌────────────────────────────────┐
    │   Logging & Analytics          │
    │   (ELK / CloudWatch)           │
    └────────────────────────────────┘
```

## Request Processing Flow

```
1. Transaction Received
   │
   ▼
2. Input Validation
   │ (Pydantic Schema)
   ▼
3. Preprocessing
   │ ├─ Feature Scaling
   │ └─ Data Transformation
   ▼
4. Model Prediction
   │ ├─ Load Model (from cache)
   │ └─ Inference
   ▼
5. Risk Scoring
   │ ├─ Probability → Risk Level
   │ └─ Threshold Application
   ▼
6. Response Generation
   │ ├─ Format JSON
   │ └─ Add Metadata
   ▼
7. Logging & Metrics
   │ ├─ Transaction Log
   │ ├─ Performance Metrics
   │ └─ Business Metrics
   ▼
8. Response Sent
```

## Data Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                  Training Pipeline                          │
└─────────────────────────────────────────────────────────────┘

Raw Data (creditcard.csv)
    │
    ▼
Data Loading & Validation
    │
    ▼
Data Preprocessing
    ├─ Handle Missing Values
    ├─ Feature Scaling (Time, Amount)
    └─ Train/Test Split (80/20)
    │
    ▼
Class Imbalance Handling
    └─ SMOTE Oversampling
    │
    ▼
Model Training
    └─ Random Forest Classifier
    │
    ▼
Model Evaluation
    ├─ Confusion Matrix
    ├─ Classification Report
    ├─ ROC-AUC Score
    └─ Visualization
    │
    ▼
Model Serialization
    ├─ Save Model (.joblib)
    └─ Save Scaler (.joblib)
    │
    ▼
Deployment
```

## Deployment Architectures

### 1. Docker Deployment (Small Scale)

```
┌─────────────────────────────────────┐
│         Docker Host                 │
│                                     │
│  ┌─────────────────────────────┐   │
│  │  fraud-detection-api        │   │
│  │  (FastAPI Container)        │   │
│  │  Port: 8000                 │   │
│  └─────────────────────────────┘   │
│                                     │
│  ┌─────────────────────────────┐   │
│  │  Redis Cache                │   │
│  │  Port: 6379                 │   │
│  └─────────────────────────────┘   │
│                                     │
│  Docker Network: fraud-detection   │
└─────────────────────────────────────┘
```

### 2. Kubernetes Deployment (Large Scale)

```
┌──────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                        │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │              Ingress Controller                     │     │
│  │              (nginx/traefik)                        │     │
│  └─────────────────────┬──────────────────────────────┘     │
│                        │                                     │
│  ┌─────────────────────┴──────────────────────────────┐     │
│  │              Service (LoadBalancer)                 │     │
│  └─────────────────────┬──────────────────────────────┘     │
│                        │                                     │
│  ┌────────────────────────────────────────────┐             │
│  │         Deployment (fraud-api)             │             │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  │             │
│  │  │ Pod1 │  │ Pod2 │  │ Pod3 │  │ PodN │  │             │
│  │  └──────┘  └──────┘  └──────┘  └──────┘  │             │
│  │                                            │             │
│  │  Horizontal Pod Autoscaler (HPA)          │             │
│  └────────────────────────────────────────────┘             │
│                        │                                     │
│  ┌────────────────────┴────────────────────┐               │
│  │           ConfigMap / Secrets            │               │
│  └──────────────────────────────────────────┘               │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ Redis StatefulSet│  │ Prometheus       │                │
│  └──────────────────┘  └──────────────────┘                │
└──────────────────────────────────────────────────────────────┘
```

## Model Training & Serving Workflow

```
Development                  Staging                 Production
─────────────               ──────────              ────────────

┌────────────┐              ┌────────────┐          ┌────────────┐
│ Data       │              │ Model      │          │ Model      │
│ Scientists │──────────────│ Validation │──────────│ Serving    │
│            │  Train       │            │  Deploy  │            │
└────────────┘              └────────────┘          └────────────┘
     │                            │                       │
     │                            │                       │
     ▼                            ▼                       ▼
┌────────────┐              ┌────────────┐          ┌────────────┐
│ Jupyter    │              │ A/B        │          │ Production │
│ Notebook   │              │ Testing    │          │ API        │
└────────────┘              └────────────┘          └────────────┘
     │                            │                       │
     │                            │                       │
     ▼                            ▼                       ▼
┌────────────┐              ┌────────────┐          ┌────────────┐
│ Training   │              │ Metrics    │          │ Monitoring │
│ Script     │              │ Analysis   │          │ & Alerts   │
└────────────┘              └────────────┘          └────────────┘
     │                            │                       │
     │                            │                       │
     ▼                            ▼                       ▼
┌────────────┐              ┌────────────┐          ┌────────────┐
│ Model      │              │ Rollback   │          │ Auto       │
│ Registry   │──────────────│ Plan       │──────────│ Scaling    │
└────────────┘              └────────────┘          └────────────┘
```

## Monitoring & Observability Stack

```
┌──────────────────────────────────────────────────────────┐
│                  Application Layer                        │
│  ┌────────────────────────────────────────────┐          │
│  │         FastAPI Application                │          │
│  │  ├─ Request Logging                        │          │
│  │  ├─ Metrics Export (Prometheus)            │          │
│  │  └─ Error Tracking                         │          │
│  └────────────┬──────────────────────┬────────┘          │
└───────────────┼──────────────────────┼───────────────────┘
                │                      │
        ┌───────┴──────┐      ┌───────┴──────┐
        ▼              ▼      ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Prometheus   │ │ Grafana      │ │ ELK Stack    │
│ (Metrics)    │ │ (Dashboards) │ │ (Logs)       │
└──────────────┘ └──────────────┘ └──────────────┘
        │              │              │
        └──────────────┴──────────────┘
                       │
                       ▼
              ┌──────────────┐
              │ Alertmanager │
              │ (Alerts)     │
              └──────────────┘
                       │
                       ▼
              ┌──────────────┐
              │ PagerDuty/   │
              │ Slack/Email  │
              └──────────────┘
```

## Security Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  Security Layers                         │
└──────────────────────────────────────────────────────────┘

Layer 1: Network Security
├─ Firewall Rules
├─ VPC/Private Networks
├─ DDoS Protection
└─ SSL/TLS Encryption

Layer 2: API Security
├─ API Key Authentication
├─ Rate Limiting
├─ Input Validation
├─ CORS Policies
└─ Request Signing

Layer 3: Application Security
├─ Secure Coding Practices
├─ Dependency Scanning
├─ Secret Management
└─ Error Handling

Layer 4: Data Security
├─ Encryption at Rest
├─ Encryption in Transit
├─ PII Anonymization
└─ Access Control (RBAC)

Layer 5: Infrastructure Security
├─ Container Scanning
├─ Image Signing
├─ Least Privilege Access
└─ Security Patching

Layer 6: Compliance & Audit
├─ Audit Logging
├─ Compliance Monitoring
├─ Incident Response
└─ Regular Security Reviews
```

## CI/CD Pipeline

```
┌─────────────┐
│ Developer   │
│ Commits     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Git Push    │
│ to Branch   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ CI Triggers │
│ (GitHub     │
│  Actions)   │
└──────┬──────┘
       │
       ├─► Lint Code (flake8, black)
       │
       ├─► Run Tests (pytest)
       │
       ├─► Security Scan (bandit, safety)
       │
       ├─► Build Docker Image
       │
       └─► Push to Registry
       │
       ▼
┌─────────────┐
│ Deploy to   │
│ Staging     │
└──────┬──────┘
       │
       ├─► Smoke Tests
       │
       ├─► Integration Tests
       │
       └─► Performance Tests
       │
       ▼
┌─────────────┐
│ Manual      │
│ Approval    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Deploy to   │
│ Production  │
└──────┬──────┘
       │
       ├─► Blue/Green Deployment
       │
       ├─► Health Checks
       │
       └─► Monitoring
       │
       ▼
┌─────────────┐
│ Production  │
│ Live        │
└─────────────┘
```
