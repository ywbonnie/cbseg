
for i in range(10):
    with open("out-{0}.txt".format(i), 'r', encoding='UTF-8') as f:
        with open("crf-results/{0}.txt".format(i), 'w', encoding='UTF-8') as outf:
            for line in f:
                if line.strip() == '':
                    outf.write('\n')
                    continue
                ss = line.strip().split('\t')
                ch = ss[0].strip()
                ty = ss[1].strip()
                outy = ss[2].strip().upper()
                ansy = ss[3].strip().upper()
                outf.write("{0}\t{1}\t{2}\t{3}\n".format(ch, ty, ansy, outy))

