import base64
from typing import Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import dh
from cryptography.hazmat.primitives.asymmetric.dh import DHPublicKey, DHParameterNumbers
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, load_pem_public_key

KEY_SIZE = 2048
GENERATOR = 2
ENCODING = Encoding.PEM
FORMAT = PublicFormat.SubjectPublicKeyInfo

class DHCryptor:

    def __init__(self, p=None, g=None):
        if p is None or g is None:
            self._parameters = dh.generate_parameters(GENERATOR, KEY_SIZE)
            pn = self._parameters.parameter_numbers()
        else:
            pn = DHParameterNumbers(p, g)
            self._parameters = pn.parameters()

        self.p = pn.p
        self.g = pn.g

        self._private_key = self._parameters.generate_private_key()
        self._public_key = self._private_key.public_key()
        self._cryptor: Union[Fernet, None] = None

    @staticmethod
    def serialize_public_key(dh_public_key: DHPublicKey) -> bytes:
        return dh_public_key.public_bytes(ENCODING, FORMAT)

    def get_public_key(self):
        """
        Return the public key used for Diffie-Helman

        Returns: public key as bytes

        """
        return self.serialize_public_key(self._public_key)

    def exchange(self, public_key: bytes) -> None:
        public_key = load_pem_public_key(public_key)
        base_key = self._private_key.exchange(public_key)
        shared_key = DHCryptor._derive_key(base_key)

        self._cryptor = Fernet(shared_key)

    @staticmethod
    def _derive_key(shared_key):
        derived = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'handshake data'
            ,
        ).derive(shared_key)

        return base64.b64encode(derived)

    def encrypt(self, data: bytes) -> bytes:
        return self._cryptor.encrypt(data)

    def decrypt(self, data: bytes) -> bytes:
        return self._cryptor.decrypt(data)
