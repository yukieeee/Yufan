# Privacy-Aware Deep Learning Architectures for the Internet of Things Applications

The training network is pre-trained using keras and tensorflow library on plain MNIST dataset, where activation functions are replaced with polynomials. Code of this part is stored in "pre-trained CNN based on Keras" folder.


The inference network encrypts the parameters of the pre-trained model using the Python wrapper of SEAL library, where necessary algorithms including Enc, Dec, KeyGen, and Eval are pre-implemented. Code of this part is stored in "Encrypted_CNN" folder.
