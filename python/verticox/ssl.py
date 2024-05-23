import datetime
import ssl
from socket import AddressFamily

import grpc
import psutil
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509 import NameOID
from vantage6.algorithm.tools.util import info

from verticox.grpc.datanode_pb2_grpc import DataNodeStub

NETWORK_INTERFACE = "eth0"
GRPC_OPTIONS = [("wait_for_ready", True)]


def get_my_ip(interface=NETWORK_INTERFACE):
    addrs = psutil.net_if_addrs()
    snic_addr = addrs[interface]

    for item in snic_addr:
        if item.family == AddressFamily.AF_INET:
            return item.address


def generate_self_signed_certificate(address):
    info(f"creating self signed certificate for address: {address}")
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    key_bytes = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "NL"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Noord-Holland"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Amsterdam"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "NLeSc"),
            x509.NameAttribute(NameOID.COMMON_NAME, address),
        ]
    )
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(
            # Our certificate will be valid for 10 days
            datetime.datetime.utcnow()
            + datetime.timedelta(days=10)
        )
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(address)]),
            critical=False,
            # Sign our certificate with our private key
        )
        .sign(key, hashes.SHA256())
    )

    cert_bytes = cert.public_bytes(encoding=serialization.Encoding.PEM)

    return key_bytes, cert_bytes


# TODO: not the right module for this function
def get_secure_stub(host, port):
    """
    Get gRPC client for the datanode.

    Args:
        ip:
        port:

    Returns:

    """
    addr = f"{host}:{port}"
    info(f"Connecting to datanode at {addr}")
    server_cert = ssl.get_server_certificate((host, port))
    credentials = grpc.ssl_channel_credentials(server_cert.encode())
    channel = grpc.secure_channel(addr, credentials=credentials, options=GRPC_OPTIONS)
    return DataNodeStub(channel)
