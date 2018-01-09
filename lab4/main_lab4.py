from lab4.RBM import RBM
from lab4.DBN import DBN
from lab4.VAE import VAE

def zad1():

    rbm = RBM()
    rbm.init_model()
    rbm.init_session()
    w, vb, hb, vr, hs = rbm.train()
    rbm.visualize(w, vb, vr, hs)

def zad2():

    dbn = DBN(100, [10, 10])
    dbn.init_model()
    dbn.init_session()
    w, vb, vr, h = dbn.train()
    dbn.visualize(w, vr, h)

def zad4():

    vae = VAE()
    vae.create_session()
    vae.init_model()
    vae.init_session()
    vae.train()
    vae.visualize()

def main():

    # zad1()
    # zad2()
    zad4()

if __name__ == "__main__":

    main()