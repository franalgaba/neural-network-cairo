import numpy as np

print("loading W1 weights...")
W1 = np.load("weights/W1.npy")
with open("w1.txt", "w") as f:
    for weight in W1.flatten().tolist():
        if weight > 0:
            sign = "false"
        else:
            sign = "true"
        addition = "tensor.append(i33 [ inner: {}_u32, sign: {} ]);\n".format(str(int(abs(weight))), sign)
        addition.replace("[", "{")
        addition.replace("]", "}")
        f.write(addition)

print("loading W2 weights...")
W2 = np.load("weights/W2.npy")
with open("w2.txt", "w") as f:
    for weight in W2.flatten().tolist():
        if weight > 0:
            sign = "false"
        else:
            sign = "true"
        addition = "tensor.append(i33 [ inner: {}_u32, sign: {} ]);\n".format(str(int(abs(weight))), sign)
        addition = addition.replace("[", "{")
        addition = addition.replace("]", "}")
        f.write(addition)

print("loading b1 biases...")
b1 = np.load("weights/b1.npy")
with open("b1.txt", "w") as f:
    for weight in b1.flatten().tolist():
        if weight > 0:
            sign = "false"
        else:
            sign = "true"
        addition = "tensor.append(i33 [ inner: {}_u32, sign: {} ]);\n".format(str(int(abs(weight))), sign)
        addition = addition.replace("[", "{")
        addition = addition.replace("]", "}")
        f.write(addition)

print("loading b2 biases...")
b2 = np.load("weights/b2.npy")
with open("b2.txt", "w") as f:
    for weight in b2.flatten().tolist():
        if weight > 0:
            sign = "false"
        else:
            sign = "true"
        addition = "tensor.append(i33 [ inner: {}_u32, sign: {} ]);\n".format(str(int(abs(weight))), sign)
        addition = addition.replace("[", "{")
        addition = addition.replace("]", "}")
        f.write(addition)