#!/usr/bin/python


def write_configs_file(file, config):
    string = ''
    for i in config:
        string += str(i) + ','
    with open(file, 'w') as write:
        write.write(string)


def read_target(target_file):
    # format a1,a2,a3...
    list_target = []
    with open(target_file) as conf_file:
        for line in conf_file:
            line = line.strip()

            # remove unexpected space
            array = line.split(',')
            for target in array:
                if len(target) > 0 and target != '\n':
                    list_target.append(float(target))
    return list_target


def parse_output(line):
    list_target = []
    line = line.strip()

    # remove unexpected space
    array = line.split(',')

    for target in array:
        try:
            if len(target) > 0 and target != '\n':
                list_target.append(float(target))
        except EOFError and IOError and ValueError:
            continue

    return list_target
