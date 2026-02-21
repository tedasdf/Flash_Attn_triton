



from im_op import im2col_triton


class CNN_layer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x, weight, padding ,stride
    ):  
        C_out , C_in , K_h, K_w = weight.shape()
        A_matrix = im2col_triton(x, K_h, K_w, stride, padding)
        
        B_matrix = 

    @staticmethod
    def backward(

    ):
        s


if __name__=="__main__":
    