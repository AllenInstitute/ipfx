"""
Wrapper class around the native struct
"""
import struct
from ipfx.x_to_nwb.hr_struct import Struct


class TreeNode(Struct):
    """Struct that also represents a node in the tree.
    """
    def __init__(self, fh, endianess, record_types, level_sizes, level=0):
        self.level = level
        self.children = []

        if self.level == 0:
            levels = struct.unpack(endianess + 'i', fh.read(4))[0]

            # read size of each level (one int per level)
            level_sizes = []
            for _ in range(levels):
                size = struct.unpack(endianess + 'i', fh.read(4))[0]
                level_sizes.append(size)

        # The record structure in the file may differ from our expected
        # structure due to version differences, so we read the required number
        # of bytes, and then pad or truncate before unpacking the record. This
        # will probably result in corrupt data in some situations..
        realsize = level_sizes[level]
        structsize = self.size()
        data = fh.read(realsize)
        diff = structsize - realsize
        if diff > 0:
            data = data + b'\0'*diff
        else:
            data = data[:structsize]

        # initialize struct data
        Struct.__init__(self, data, endianess)

        # next read the number of children
        nchild = struct.unpack(endianess + 'i', fh.read(4))[0]

        level += 1
        if level >= len(record_types):
            return
        child_rectype = record_types[level]
        for _ in range(nchild):
            self.children.append(child_rectype(fh, endianess, record_types,
                                 level_sizes, level))

    def __getitem__(self, i):
        return self.children[i]

    def __len__(self):
        return len(self.children)

    def __iter__(self):
        return self.children.__iter__()

    def __str__(self, indent=0):
        # Return a string describing this structure
        ind = '    '*indent
        srep = Struct.__str__(self, indent)[:-1]  # exclude final parentheses
        srep += ind
        srep += '    children = %d,\n' % len(self)
        srep += ind
        srep += ')'
        return srep
