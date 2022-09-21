IMAGE=harbor.carrier-mu.src.surf-hosted.nl/carrier/verticox_predictors

docker build --no-cache --build-arg SSH_PRIVATE_KEY="$(cat C:\\Florian\\GIT\\KEY\\id_rsa)" \
  --build-arg SSH-PUBLIC-KEY="$(cat C:\\Florian\\GIT\\KEY\\id_rsa.pub)" \
  -t verticox_predictor  .

docker tag verticox_predictor $IMAGE

docker push $IMAGE