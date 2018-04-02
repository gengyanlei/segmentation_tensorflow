# leilei
HED tensorflow

声明：
tensorflow 1.2.1 及其之后版本均可以使用。
这里我们参考了HED的网络结构，由于HED网络是在caffe框架下实现的，caffe安装不易
因此，我们在tensorflow框架下实现它，并且做了一定的修改结构，但是思想没有变。仍然是concat巻积层特征。

需要使用 VGG16 weights 进行 fine-tune.
VGG16权重保存为npy格式，较之tensorflow的ckpt会更灵活些。
