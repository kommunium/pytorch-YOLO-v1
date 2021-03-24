# %%
path = "labels/"
with open('label_train.txt', 'w') as fw:
    for i in range(1, 201):
        file_name = str(i).rjust(7, '0')
        data = file_name + '.jpg '
        with open(path + file_name + '.txt', 'r') as fr:
            for line in fr.readlines():
                obj = line.strip().split(' ')
                c = obj[0]
                obj = list(map(float, obj[1:]))
                x1 = round((obj[0] - .5 * obj[2]) * 640)
                y1 = round((obj[1] - .5 * obj[3]) * 480)
                x2 = round((obj[0] + .5 * obj[2]) * 640)
                y2 = round((obj[1] + .5 * obj[3]) * 480)
                data += '{} {} {} {} {} '.format(x1, y1, x2, y2, c)
            fw.write(data.strip() + '\n')

# %%
with open('label_train.txt', 'w') as fw:
    for i in range(1, 201):
        file_name = path + str(i).rjust(7, '0') + '.txt'
        with open(file_name, 'r') as fr:
            for line in fr.readlines():
                fw.write(line.strip() + ' ')
            fw.write('\n')
