

import numpy as np

cifar10_alexnet_ALSH_times = np.array([1.81, 1.86, 1.74, 1.87, 1.89])
cifar100_alexnet_ALSH_times = np.array([1.87, 1.87, 1.78, 2.00, 1.98])

cifar10_alexnet_default_times = np.array([2.93, 2.95, 2.94, 2.94, 2.94])
cifar100_alexnet_default_times = np.array([2.94, 2.94, 2.94, 2.94, 2.95])

print('ALSH alexnet avg and std-dev: ')
print(np.average(cifar10_alexnet_ALSH_times))
print(np.std(cifar10_alexnet_ALSH_times))
print(np.average(cifar100_alexnet_ALSH_times))
print(np.std(cifar100_alexnet_ALSH_times))

print()
print('Default alexnet avg and std-dev: ')
print(np.average(cifar10_alexnet_default_times))
print(np.std(cifar10_alexnet_default_times))
print(np.average(cifar100_alexnet_default_times))
print(np.std(cifar100_alexnet_default_times))


print('\n\n')
cifar10_vgg11_ALSH_times = np.array([18.88, 17.32, 18.41, 19.50, 18.57])
cifar100_vgg11_ALSH_times = np.array([17.2, 18.61, 18.55, 17.75, 18.80])

cifar10_vgg11_default_times = np.array([25.49, 25.24, 25.24, 25.23, 25.43])
cifar100_vgg11_default_times = np.array([25.23, 25.23, 25.25, 25.34, 25.29])

print('ALSH vgg11 avg and std-dev: ')
print(np.average(cifar10_vgg11_ALSH_times))
print(np.std(cifar10_vgg11_ALSH_times))
print(np.average(cifar100_vgg11_ALSH_times))
print(np.std(cifar100_vgg11_ALSH_times))

print()
print('Default vgg11 avg and std-dev: ')
print(np.average(cifar10_vgg11_default_times))
print(np.std(cifar10_vgg11_default_times))
print(np.average(cifar100_vgg11_default_times))
print(np.std(cifar100_vgg11_default_times))
