import math
import torch


def rotate_coord(coord, angle):
    angle = float(angle)
    rad = torch.tensor(-angle * math.pi / 180)
    c = torch.cos(rad)
    s = torch.sin(rad)
    rotation_matrix = torch.stack([torch.stack([c, -s]),
                                    torch.stack([s, c])])
    
    shifting = torch.full(coord.shape, -383/2)
    coord = coord + shifting
    new_coord = coord @ rotation_matrix.t()
    new_coord = new_coord - shifting
    return new_coord