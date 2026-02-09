import mixture_mi
import mi_mao



def NoMixtureNoClusteredMI(discs, conts):

    cnmi = mi_mao.NMI(discs, conts)

    return cnmi

def NoClusteredMI(discs, conts):

    cnmi = mixture_mi.NMI(discs, conts)

    return cnmi