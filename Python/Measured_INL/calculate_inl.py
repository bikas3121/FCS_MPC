import csv
import matplotlib.pyplot as plt
import numpy as np

def calculate_inl(measurement_file):
    headings = []
    level = []
    measured_voltage = []

    with open(measurement_file, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0].startswith('#'):
                continue
            if not headings:
                headings = row
                continue
            level.append(int(row[1]))
            measured_voltage.append(float(row[3]))

    # Sort the codes and voltages
    measured_voltage = [x for _,x in sorted(zip(level,measured_voltage))]
    level = sorted(level)
    ideal_voltage = np.linspace(measured_voltage[0], measured_voltage[-1], num=len(level))

    deviation_in_v = ideal_voltage - measured_voltage
    v_per_lsb = (measured_voltage[-1] - measured_voltage[0]) / len(level)
    deviation_in_lsb = deviation_in_v / v_per_lsb

    DNL = [0] * len(level)
    for index in range(1, len(level)):
        DNL[index] = 1 - ((measured_voltage[index] - measured_voltage[index-1]) / v_per_lsb)

    INL = deviation_in_lsb

    # plt.plot(level, ideal_voltage)
    # plt.plot(level, measured_voltage)
    # plt.title('Ideal vs measured voltage')
    # plt.xlabel('Code')
    # plt.ylabel('Voltage')
    # plt.show()

    return(INL, DNL)

#[INL, DNL] = calculate_inl("measurements_2023-10-08.csv")
# [INL, DNL] = calculate_inl("measurement_11Oct23.csv")

# # plt.plot(level, ideal_voltage, measured_voltage)
# # plt.title('Ideal vs measured voltage')
# # plt.xlabel('Code')
# # plt.ylabel('Voltage')
# # plt.show()

# plt.plot(range(len(INL)), INL)
# plt.title('INL in LSB')
# plt.xlabel('Code')
# plt.ylabel('LSB')
# plt.show()

# plt.plot(range(len(DNL)), DNL)
# plt.title('DNL in LSB')
# plt.xlabel('Code')
# plt.ylabel('LSB')
# plt.show()
