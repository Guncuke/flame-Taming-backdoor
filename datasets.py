from torchvision import datasets, transforms


def get_dataset(dir_name, name):

	if name == 'mnist':
		transform_train = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))])
		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))])
		train_dataset = datasets.MNIST(dir_name, train=True, download=False, transform=transform_train)
		eval_dataset = datasets.MNIST(dir_name, train=False, download=False, transform=transform_test)

	elif name == 'fmnist':
		transform_train = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.2860,), (0.3520,))])
		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.2860,), (0.3520,))])
		train_dataset = datasets.FashionMNIST(dir_name, train=True, download=False, transform=transform_train)
		eval_dataset = datasets.FashionMNIST(dir_name, train=False, download=False, transform=transform_test)

	elif name == 'cifar':
		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		train_dataset = datasets.CIFAR10(dir_name, train=True, download=False, transform=transform_train)
		eval_dataset = datasets.CIFAR10(dir_name, train=False, download=False, transform=transform_test)

	return train_dataset, eval_dataset

