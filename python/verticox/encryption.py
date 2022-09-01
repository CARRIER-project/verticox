import datetime

from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def create_self_signed_certificate(common_name: str):
    key = _generate_rsa_key()

    # Various details about who we are. For a self-signed certificate the
    # subject and issuer are always the same.
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, u"NL"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"Noord-Holland"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, u"Amsterdam"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"the Netherlands eScience Center"),
        x509.NameAttribute(NameOID.COMMON_NAME, u"common_name"),
    ])
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.utcnow()
    ).not_valid_after(
        # Our certificate will be valid for 10 days
        datetime.datetime.utcnow() + datetime.timedelta(days=10)
    ).add_extension(
        x509.SubjectAlternativeName([x509.DNSName(u"localhost")]),
        critical=False,
        # Sign our certificate with our private key
    ).sign(key, hashes.SHA256())

    return key, cert


def _generate_rsa_key():
    key = rsa.generate_private_key(public_exponent=65537,
                                   key_size=2048)
    return key


def generate_rsa_private_key():
    key = _generate_rsa_key()

    return key.private_bytes(encoding=serialization.Encoding.PEM,
                             format=serialization.PrivateFormat.TraditionalOpenSSL,
                             encryption_algorithm=serialization.NoEncryption())


def create_server_credentials(host: str):
    key, certificate = create_self_signed_certificate(host)

    key_bytes = key.private_bytes(encoding=serialization.Encoding.PEM,
                                  format=serialization.PrivateFormat.TraditionalOpenSSL,
                                  encryption_algorithm=serialization.NoEncryption())

    certificate_bytes = certificate.public_bytes(serialization.Encoding.PEM)

    return key_bytes, certificate_bytes
