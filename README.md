# Can
Creative adversarial network. similar to a generative adversarial network with a few key differences being the generator model is an attention transformer and that aswell a discriminator to classify images as being real or fake thus enforcing realistic image generation it also trains using loss from an image classifier trained to predict what group the image belongs to, the generator tries to produce images regarded as real by the discriminator and that yield an ambigous class label from the classifying network thus containing all the features in the image groups.
