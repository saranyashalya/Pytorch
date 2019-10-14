import torch

class MyRelu(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        print(ctx.saved_tensors)
        grad_input = grad_output.clone()
        grad_input[input<0] = 0
        return grad_input


dtype = torch.float
device = torch.device("cpu")
#if gpu
#device = torch.devie("cuda:0")

#initializing parameters (batch_size, input dimension, hidden size, output dimension
N , D_in, H, D_out = 64, 1000, 100, 10

#gerating random inputs
x = torch.randn(N,D_in, dtype = dtype, device = device)
y = torch.randn(N, D_out, dtype = dtype, device = device)

#initializing random weights for 2 layer network
w1 = torch.randn(D_in, H, dtype = dtype, device = device, requires_grad= True)
w2 = torch.randn(H, D_out, dtype = dtype, device =device, requires_grad= True)

learning_rate = 1e-6
#forward propagation
for t in range(500):
    relu = MyRelu.apply
    #calculating y_pre
    y_pred = relu(x.mm(w1)).mm(w2)

    #calculating loss
    loss = (y_pred-y).pow(2).sum()
    if t % 100 == 99:
        print("loss at %.3f is %.3f" %(t,loss.item()))

    #backward propagation
    loss.backward()

    # manually updating weights
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        #manually zero the gradients after upating weights
        w1.grad.zero_()
        w2.grad.zero_()
