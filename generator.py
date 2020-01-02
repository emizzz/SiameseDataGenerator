class SiameseDataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, n_classes=10, batch_size=1, shuffle=False):
        x, y = data
        digit_indices = [np.where(y == i)[0] for i in range(n_classes)]
        
        self.shuffle = shuffle
        self.pairs, self.y = self.create_pairs(x, digit_indices)
        self.pairs_0 = self.pairs[:, 0]
        self.pairs_1 = self.pairs[:, 1]

        self.batch_size = batch_size
        self.samples_per_train  = (self.pairs.shape[0]/self.batch_size)*self.batch_size

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.samples_per_train / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            rand_idx = np.random.permutation(self.pairs.shape[0])
            self.pairs, self.y = self.pairs[rand_idx, :, :, :, :], self.y[rand_idx]
            self.pairs_0 = self.pairs[:, 0]
            self.pairs_1 = self.pairs[:, 1]

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate data

        b_from = index*self.batch_size      #batch start index
        b_to = (index+1)*self.batch_size    #batch end index
        return self.__data_generation(b_from, b_to)

    def __data_generation(self, b_from, b_to):
        return ([   self.pairs_0[b_from: b_to], 
                    self.pairs_1[b_from: b_to]
                ],
                self.y[b_from: b_to]
            )

    def create_pairs(self, x, digit_indices):
      '''Positive and negative pair creation.
      Alternates between positive and negative pairs.
      '''
      pairs = []
      labels = []
      n = min([len(digit_indices[d]) for d in range(n_classes)]) - 1

      for d in range(n_classes):  
          for i in range(n):
              z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
              pairs += [[x[z1], x[z2]]]
              inc = random.randrange(1, n_classes)
              dn = (d + inc) % n_classes
              z1, z2 = digit_indices[d][i], digit_indices[dn][i]
              pairs += [[x[z1], x[z2]]]
              labels += [1, 0]

      return np.array(pairs), np.array(labels)
