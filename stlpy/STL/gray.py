import numpy as np

def gray2binary(n):
    n = int(n, 2)
    mask = n
    while mask:
        mask >>= 1
        n ^= mask
    return bin(n)[2:].zfill(4)

class GrayCode:
    def __init__(self):
        self.base = ["0", "1"]

    def getNext(self , prelist,z_or_one):
        output = []
        for code in prelist:
            new_code = "%s%s" % (z_or_one,code)
            output.append(new_code)
        if z_or_one == 1:
            output.reverse()

        return output

    def gray(self):
        haf1 = self.getNext(self.base, 0)
        haf2 = self.getNext(self.base, 1)
        ret = haf1 + haf2
        self.base = ret

    def getGray(self, n):
        for i in range(n-1):
            self.gray()
        return self.base














