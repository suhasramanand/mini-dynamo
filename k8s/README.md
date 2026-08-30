# Kubernetes deployment

Manifests to run Mini-Dynamo on a local Kubernetes cluster (kind, minikube, or
Docker Desktop's built-in Kubernetes). Everything is namespaced under
`mini-dynamo`.

Prefill and decode run as **StatefulSets** with headless Services so each pod
gets a stable DNS identity (`decode-0.decode`, `decode-1.decode`, ...). Each
pod advertises that identity in the Redis registry, which is what enables
KV-cache-aware sticky routing and fallback.

## Prerequisites

- A running cluster and `kubectl` pointed at it.
- The application image available to the cluster's nodes.

## 1. Build and load the image

The manifests use `image: mini-dynamo:latest` with `imagePullPolicy: IfNotPresent`,
so the image must exist on the cluster nodes.

```bash
# Build (from the repository root)
docker build -t mini-dynamo:latest .

# Load it into the cluster
kind load docker-image mini-dynamo:latest          # kind
# minikube image load mini-dynamo:latest           # minikube
# (Docker Desktop Kubernetes shares the local Docker images automatically)
```

## 2. Deploy

```bash
kubectl apply -k k8s/
kubectl -n mini-dynamo rollout status deploy/router
kubectl -n mini-dynamo get pods
```

## 3. Access the services

```bash
kubectl -n mini-dynamo port-forward svc/router 8000:8000 &
kubectl -n mini-dynamo port-forward svc/grafana 3000:3000 &
kubectl -n mini-dynamo port-forward svc/jaeger 16686:16686 &
kubectl -n mini-dynamo port-forward svc/prometheus 9090:9090 &

curl -N -X POST localhost:8000/v1/generate \
  -H 'content-type: application/json' \
  -d '{"session_id":"s1","prompt":"hello from kubernetes","max_tokens":32}'
```

- Router API: http://localhost:8000
- Grafana: http://localhost:3000 (dashboard "Mini-Dynamo Overview")
- Jaeger: http://localhost:16686
- Prometheus: http://localhost:9090

## 4. Scale independently

Prefill and decode scale on their own — the point of disaggregation:

```bash
kubectl -n mini-dynamo scale statefulset/decode --replicas=3
kubectl -n mini-dynamo scale statefulset/prefill --replicas=2
```

New pods self-register and the router begins routing to them automatically.

## 5. Tear down

```bash
kubectl delete -k k8s/
```

## Configuration

Shared settings live in the `mini-dynamo-config` ConfigMap
([`config.yaml`](config.yaml)); per-role settings (memory size, eviction
policy, node identity) are set in each workload's manifest.
