import matplotlib.pyplot as plt
def show_created_pic(generator, x):
    y=generator(x)
    y=y.numpy()
    plt.imshow(y.numpy().reshape(28, 28) / 255 - 0.5, 'gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return