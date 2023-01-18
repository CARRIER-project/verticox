from verticox.crypto import DHCryptor

MESSAGE = 'my name is rumpelstiltskin'.encode('UTF-8')

def test_public_keys_different():
    my_cryptor = DHCryptor()
    peer_cryptor = DHCryptor(my_cryptor.p, my_cryptor.g)

    assert my_cryptor.get_public_key() != peer_cryptor.get_public_key()


def test_peer_can_read_encrypted_data():
    my_cryptor = DHCryptor()
    peer_cryptor = DHCryptor(my_cryptor.p, my_cryptor.g)

    my_cryptor.exchange(peer_cryptor.get_public_key())
    peer_cryptor.exchange(my_cryptor.get_public_key())

    ciphertext = my_cryptor.encrypt(MESSAGE)
    print(ciphertext)
    decrypted = peer_cryptor.decrypt(ciphertext)

    assert decrypted == MESSAGE

def test_read_own_encrypted_data():
    my_cryptor = DHCryptor()
    peer_cryptor = DHCryptor(my_cryptor.p, my_cryptor.g)

    my_cryptor.exchange(peer_cryptor.get_public_key())

    ciphertext = my_cryptor.encrypt(MESSAGE)
    print(ciphertext)
    decrypted = my_cryptor.decrypt(ciphertext)

    assert decrypted == MESSAGE