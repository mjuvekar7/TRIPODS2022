# import NET
import HELP
import TNET

net = TNET.TNetwork([2, 8, 1], "Sigmoid", "Sigmoid", 3)
net.initialize(-1, 1)

trains = HELP.getTrains(2, ["XOR", "OR", "AND"])
x_trains = trains[0]
y_trains = trains[1]
x_tests = trains[0]
y_tests = trains[1]

net.ttp(x_trains=x_trains, y_trains=y_trains, loss_function="MSE", learn_rates=[0.1, 0.1, 0.1], num_times_list=[400, 200, 600], x_tests=x_tests, y_tests=y_tests, thresh=0.5)
