from . import *


class Decoder_Diffusion(nn.Module):
	'''
	Decode the encoded image and get message
	'''

	def __init__(self, H, W, message_length, blocks=4, channels=64, diffusion_length=256):
		super(Decoder_Diffusion, self).__init__()

		stride_blocks = int(np.log2(H // int(np.sqrt(diffusion_length))))

		self.diffusion_length = diffusion_length
		self.diffusion_size = int(self.diffusion_length ** 0.5)

		self.first_layers = nn.Sequential(
			ConvBNRelu(12, channels),
			SENet_decoder(channels, channels, blocks=stride_blocks + 1),
			ConvBNRelu(channels * (2 ** stride_blocks), channels),
		)
		self.keep_layers = SENet(channels, channels, blocks=1)
		self.final_layer = ConvBNRelu(channels, 1)
		self.adp = nn.AdaptiveAvgPool2d((self.diffusion_size, self.diffusion_size))

		self.message_layer = nn.Linear(self.diffusion_length, message_length)

	def forward(self, noised_image):
		# print(f'noised_image.shape is {noised_image.shape}')
		x = self.first_layers(noised_image)
		# print(f'x.shape: {x.shape}')
		x = self.keep_layers(x)
		# print(f'x2.shape: {x.shape}')
		x = self.final_layer(x)
		# print(f'x3.shape: {x.shape}')
		x = self.adp(x)
		# print(f'x_adp.shape: {x.shape}')
		x = x.view(x.shape[0], -1)
		# print(f'x4.shape: {x.shape}')

		x = self.message_layer(x)
		# print(f'x_final.shape: {x.shape}')
		return x
