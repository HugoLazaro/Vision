
class BrainImageSet(Dataset):
    """ Brain image set """
    def __init__(self, image_path, label_path='', deploy=False):
        
        # Initialize self variables
        ### Insert your code ###
        self.image_path = image_path
        self.deploy = deploy
        self.images = []
        self.labels = []
        ### End of your code ###
        
        image_names = sorted(os.listdir(image_path))
        for image_name in image_names:
            # Read the image. Use imageio.imread.
            ### Insert your code ###
            image = imageio.imread(os.path.join(image_path, image_name))
            self.images.append(image)
            ### End of your code ###
            
            self.images += [image]

            # Read the label map
            if not self.deploy:
                ### Insert your code ###
                label_name = image_name.replace('t1ce', 'seg')
                label = imageio.imread(os.path.join(label_path, label_name))
                self.labels.append(label)
                ### End of your code ###
        

    def __len__(self): # Number of images
        ### Insert your code ###
        return len(self.images)
        ### End of your code ###
        

    def __getitem__(self, idx):
        # Get an image and perform intensity normalisation
        # Dimension: XY
        ### Insert your code ###
        image = self.images[idx]
        ### End of your code ###
        
        # Get its label map
        # Dimension: XY
        ### Insert your code ###
        if not self.deploy:
            label = self.labels[idx]
            return image, label
        else:
            return image
        ### End of your code ###
        
        #return image, label

    
    def get_random_batch(self, batch_size):
        # Get a batch of paired images and label maps
        # Dimension of images: NCXY
        # Dimension of labels: NXY
        images, labels = [], []

        ### Insert your code ###
        for i in range(batch_size):
          # Choose a random index
          idx = random.randint(0, len(self.labels)-1)
          
          # Get the image and label at the index
          image = self.images[idx]
          label = self.labels[idx]

          # Append to the lists
          images.append(image)
          labels.append(label)
        ### End of your code ###
        
        images = np.array( images )
        print(images.shape)
        
        images = np.expand_dims(images, axis = 1) #NCXY dimension, with C = 1

        labels = np.array( labels )

        return images, labels
    