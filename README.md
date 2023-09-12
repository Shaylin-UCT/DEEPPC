# 1hash
## 2hash
### 3hash
#### 4hash
##### 5hash
Reference code: [test](./Data_to_PyDataset.py)
## Data_to_PyDataset.py
The [Data_to_PyDataset.py](./Data_to_PyDataset.py) is responsible for converting a folder of images into a dataset usable by PyTorch by extending the built-in Dataset class. The main method is included as a test case. Instantiating the class requires 2 parameters: the image size and the *element*. The element is the class as defined above. It will return an instance of *torch.utils.data.Dataset*. The main method includes a test case that will print "DATALOADER IS CORRECT" if the code is set up correctly as well as an example of an image from the dataset. 

In the GAN, instantiate the class and call *getData()*. The returned object should be passed to the Dataloader within respective *dataset* parameter. 