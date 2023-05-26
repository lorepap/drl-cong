import os
from enum import Enum
from helper import context


class MahimahiTrace(Enum):

    att_lte_driving = 'att.lte.driving'
    att_lte_driving_2016 = 'att.lte.driving.2016'
    tm_lte_driving = 'tm.lte.driving'
    tm_lte_short = 'tm.lte.short'
    tm_umts_driving = 'tm.umts.driving'
    vz_evdo_driving = 'vz.evdo.driving'
    vz_lte_driving = 'vz.lte.driving'
    vz_lte_short = 'vz.lte.short'
    bus = 'bus'
    timessquare = 'timessquare'
    wired = 'wired'
    none = 'none'

    def fromString(trace: str):
        return MahimahiTrace[trace.replace('.', '_')]

    def path(self):
        paths = []
        dir = os.path.join(context.src_dir, 'traces')

        if self == MahimahiTrace.att_lte_driving:
            paths += [os.path.join(dir, 'ATT-LTE-driving.up'),
                      os.path.join(dir, 'ATT-LTE-driving.down'),
                      ]

        elif self == MahimahiTrace.att_lte_driving_2016:
            paths += [os.path.join(dir, 'ATT-LTE-driving-2016.up'),
                      os.path.join(dir, 'ATT-LTE-driving-2016.down'),
                      ]

        elif self == MahimahiTrace.tm_lte_driving:
            paths += [os.path.join(dir, 'TMobile-LTE-driving.up'),
                      os.path.join(dir, 'TMobile-LTE-driving.down'),
                      ]

        elif self == MahimahiTrace.tm_lte_short:
            paths += [os.path.join(dir, 'TMobile-LTE-short.up'),
                      os.path.join(dir, 'TMobile-LTE-short.down'),
                      ]

        elif self == MahimahiTrace.tm_umts_driving:
            paths += [os.path.join(dir, 'TMobile-UMTS-driving.up'),
                      os.path.join(dir, 'TMobile-UMTS-driving.down'),
                      ]

        elif self == MahimahiTrace.vz_evdo_driving:
            paths += [os.path.join(dir, 'Verizon-EVDO-driving.up'),
                      os.path.join(dir, 'Verizon-EVDO-driving.down'),
                      ]

        elif self == MahimahiTrace.vz_lte_driving:
            paths += [os.path.join(dir, 'Verizon-LTE-driving.up'),
                      os.path.join(dir, 'Verizon-LTE-driving.down'),
                      ]

        elif self == MahimahiTrace.vz_lte_short:
            paths += [os.path.join(dir, 'Verizon-LTE-short.up'),
                      os.path.join(dir, 'Verizon-LTE-short.down'),
                      ]
        
        elif self == MahimahiTrace.bus:
            paths += [os.path.join(dir, 'wired48'),
                      os.path.join(dir, 'trace-3109898-bus'),
                      ]

        elif self == MahimahiTrace.timessquare:
            paths += [os.path.join(dir, 'wired48'),
                      os.path.join(dir, 'trace-3189663-timessquare'),
                      ]
        
        elif self == MahimahiTrace.wired:
            paths += [os.path.join(dir, 'wired48'),
                      os.path.join(dir, 'wired6'),
                      ]

        else:
            paths += ['', '']

        return paths[0], paths[1]
