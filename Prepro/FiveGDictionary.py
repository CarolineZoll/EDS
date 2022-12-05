
class ModulationCodingScheme:

    mod_dict = \
    {
        0:(2,120),
        1:(2,193),
        2:(2,308),
        3:(2,449),
        4:(2,602),
        5:(4,378),
        6:(4,434),
        7:(4,490),
        8:(4,553),
        9:(4,616),
        10:(4,658),
        11:(6,466),
        12:(6,517),
        13:(6,567),
        14:(6,616),
        15:(6,666),
        16:(6,719),
        17:(6,772),
        18:(6,822),
        19:(6,873),
        20:(8,682.5),
        21:(8,711),
        22:(8,754),
        23:(8,797),
        24:(8,841),
        25:(8,885),
        26:(8,916.5),
        27:(8,948),

    }

    def __init__(self,mcs_index):
        self.mcs_index = mcs_index

    def getModOrder(self):
        res = self.mod_dict[self.mcs_index] #get value from the mod_dict of the given mcs_index
        return res[0]

    def getCodeRate(self):
        res = self.mod_dict[self.mcs_index]  # get value from the mod_dict of the given mcs_index
        return res[1]


class ScalingFactor:
    scal_dict = \
        {
            (4,8):1,
            (2,8):0.5,
            (4,6):0.75,
            (1,8):1,#this could be wrong, but there aren't any layer layouts with only 1 configuration what is the scale then?
            (1, 6): 1,
            (1, 4): 1,
            (1, 2): 1
        }

    def __init__(self,layer_num: int, mod_order: int):
        self.key = (layer_num,mod_order)

    def get_scale_factor(self):
        res = self.scal_dict[self.key]  # get value from the mod_dict of the given mcs_index
        return res

class StandardizedValues:

    band_values_dict = \
    {
        #current known band values
        "LTE-1800": 1800,
        "LTE-900": 900,
        "LTE-2600": 2600,
        "LTE-800":800,
        "LTE-2100":2100,
        "Unknown": 0,
        "GSM":0,
        "LTE-700":700
    }

class HeaderTags:

    tag_dict = \
    {
        ##Telekom Standardization
        'lte_rssnr': 'rssnr',
        'band': 'band',#needs to be spit into atomix parts
        'lte_cqi': 'cqi',
        'lon':'lon',
        'lat':'lat',
        'hw_model':'model',#HFapp Standardization
        'model':'model',
        'rssnr':'rssnr',
        'cqi':'cqi',
        'datetime':'time',
        'time':'time'
    }

    def __init__(self,tag):
        self.key = tag

    def get_standardized_tag(self):
        key_exists = self.key in self.tag_dict

        if key_exists:
            return self.tag_dict[self.key]
        else:
            return "None"


