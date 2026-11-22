# i will use numpy for the implementation
import numpy as np
import math

# The implementation
class SelfAttention():
    def __init__(self , hidden_size:int , input_size:int):
        """
        This class attention get two parameters :\n
        hidden_size : the size that input will processed in it \n 
        input_size : the last size of input dimention\n
        *The result is (seq , hidden_size)*
        """

        # first define the Hyper-paramerters
        self.input_size = input_size
        self.hidden_size = hidden_size

        # then define the layer all the layer will have the same dimension 
        self.query = np.random.randn(self.input_size , self.hidden_size) * 1/input_size
        self.key = np.random.randn(self.input_size , self.hidden_size) * 1/hidden_size
        self.value = np.random.randn(self.input_size , self.hidden_size) * 1/hidden_size



    def forward(self , input:np.array) -> np.array:
        query = np.dot(input , self.query)
        key = np.dot(input , self.key)
        value = np.dot(input , self.value)

        # equation => softmax((q * k.T)/sqrt(dim)) * value  => (*) is dot product

        q_k = np.dot(query , key.T)
        scaled_qk = q_k / math.sqrt(self.hidden_size)
        
        attention_scores = Softmax(scaled_qk)
        output = np.dot(attention_scores , value)

        return output , attention_scores 



def Softmax(input : np.array) -> np.array :
    # equation => e^x / sum (1 - n) (e^x)

    output = np.e ** input
    output /= np.sum(output , axis=1 , keepdims=True)


    return output 



# example
input = np.random.randn( 25 , 512)

selfattention = SelfAttention(512,512)

out , attention_scores = selfattention.forward(input)

print(f"The output shape is : \n{out.shape}")
print(f"{"="*50}")
print(f"The output is : \n{out}")
print(f"{"="*50}")
print(f"The attention scores is : \n{attention_scores}")

# ensure that the sum of all probability is (1)
print(f"{"=" *50}")

print(f"The sum of all probability : \n{np.sum(attention_scores,axis=1 , keepdims=True)}")

