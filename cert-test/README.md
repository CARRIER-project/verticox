# Grpc certificate test
The first step will be to only test with server certificates, not client certificates.
When GRPC is working with server certificates, we will add in the client certificates.

It still remains to be seen how these certificates will be distributed using vantage6.
Ideally these certificates need to be handled by the vantage6 server


## Components
### CA
Docker container that serves as CA. Has root certificate that needs to be distributed to other containers

### Datanode container
Container that runs the DataNode server. Needs to have a certificate issued by the CA container.
It needs to have the CA root certificate in its list of trusted certificates.

### Client container
Container that will call the Grpc Datanode. It also needs to have the root certificate in its list of
trusted certificates.
