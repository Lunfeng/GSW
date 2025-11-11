from . import *


class Encoder_MP_Diffusion(nn.Module):
	'''
	Insert a watermark into an image
	'''

	def __init__(self, H, W, message_length, blocks=4, channels=64, diffusion_length=256):
		super(Encoder_MP_Diffusion, self).__init__()
		self.diffusion_length = diffusion_length
		self.diffusion_size = int(diffusion_length ** 0.5)
		stride_blocks = int(np.log2(H // int(np.sqrt(diffusion_length))))

		self.image_pre_layer = ConvBNRelu(12, channels)
		# self.image_pre_layer = ConvBNRelu(3, channels)
		self.image_first_layer = SENet(channels, channels, blocks=blocks)

		self.message_duplicate_layer = nn.Linear(message_length, self.diffusion_length)
		self.message_pre_layer_0 = ConvBNRelu(1, channels)
		self.message_pre_layer_1 = ExpandNet(channels, channels, blocks=stride_blocks)
		self.upsample = nn.Upsample(size=(H, W), mode='bilinear', align_corners=True)
		self.message_pre_layer_2 = SENet(channels, channels, blocks=1)
		self.message_first_layer = SENet(channels, channels, blocks=blocks)

		self.after_concat_layer = ConvBNRelu(2 * channels, channels)

		self.final_layer = nn.Conv2d(channels + 12, 12, kernel_size=1)

	def forward(self, image, message):
		# print(f'Encoder_MP input image.shape is {image.shape}')
		# print(f'Encoder_MP input message.shape is {message.shape}')
		# first Conv part of Encoder
		image_pre = self.image_pre_layer(image)
		# print(f'Encoder_MP input image_pre.shape is {image_pre.shape}')
		intermediate1 = self.image_first_layer(image_pre)
		# print(f'Encoder_MP input intermediate1.shape is {intermediate1.shape}')
		# Message Processor (with diffusion)
		message_duplicate = self.message_duplicate_layer(message)
		# print(f'Encoder_MP input message_duplicate.shape is {message_duplicate.shape}')
		message_image = message_duplicate.view(-1, 1, self.diffusion_size, self.diffusion_size)
		# print(f'Encoder_MP input message_image.shape is {message_image.shape}')
		message_pre_0 = self.message_pre_layer_0(message_image)
		# print(f'Encoder_MP input message_pre_0.shape is {message_pre_0.shape}')
		message_pre_1 = self.message_pre_layer_1(message_pre_0)

		message_pre_upsample = self.upsample(message_pre_1)
		message_pre_2 = self.message_pre_layer_2(message_pre_upsample)
		# print(f'Encoder_MP input message_pre_layer_2.shape is {message_pre_2.shape}')
		intermediate2 = self.message_first_layer(message_pre_2)
		# print(f'Encoder_MP input intermediate2.shape is {intermediate2.shape}')

		# concatenate
		concat1 = torch.cat([intermediate1, intermediate2], dim=1)
		# print(f'Encoder_MP input concat1.shape is {concat1.shape}')
		# second Conv part of Encoder
		intermediate3 = self.after_concat_layer(concat1)
		# print(f'Encoder_MP input intermediate3.shape is {intermediate3.shape}')
		# skip connection
		concat2 = torch.cat([intermediate3, image], dim=1)
		# print(f'Encoder_MP input concat2.shape is {concat2.shape}')

		# last Conv part of Network
		output = self.final_layer(concat2)
		# print(f'Encoder_MP input output.shape is {output.shape}')

		return output