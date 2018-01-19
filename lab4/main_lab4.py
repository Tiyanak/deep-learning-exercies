from lab4.RBM import RBM
from lab4.DBN import DBN
from lab4.VAE import VAE
from lab4.DBN_Fine_Tune import DBN_Fine_Tune

def zad1():

    # zad 1
    rbm = RBM()
    rbm.init_model()
    rbm.init_session()
    w, vb, hb, vr, hs = rbm.train()
    rbm.visualize(w, vb, vr, hs)

def zad2():

    rbm = RBM()
    rbm.init_model()
    rbm.init_session()
    w, vb, hb, vr, hs = rbm.train()
    rbm.visualize(w, vb, vr, hs)

    dbn = DBN()
    dbn.init_model(w, vb, hb)
    dbn.init_session()
    w1, w2, v_bias, h1_bias, h2_bias, v_out_prob, h_top = dbn.train()
    dbn.visualize(w1, w2, v_bias, h1_bias, h2_bias, v_out_prob, h_top)

def zad3():

    rbm = RBM()
    rbm.init_model()
    rbm.init_session()
    w1, v_bias, h1_bias_up, v_out_prob1, h_top = rbm.train()
    rbm.visualize(w1, v_bias, v_out_prob1, h_top)

    dbn = DBN()
    dbn.init_model(w1, v_bias, h1_bias_up)
    dbn.init_session()
    w1, w2, v_bias, h1_bias_down, h2_bias, v_out_prob2, h_top = dbn.train()
    dbn.visualize(w1, w2, v_bias, h1_bias_down, h2_bias, v_out_prob2, h_top)

    dbnFT = DBN_Fine_Tune()
    dbnFT.init_model(w1, w2, v_bias, h1_bias_up, h1_bias_down, h2_bias)
    dbnFT.init_session()
    R1, W1, W2, v_bias, h1_bias_up, h1_bias_down, h2_bias, v_out_prob3, h_top, h_top_prob = dbnFT.train()
    dbnFT.visualize(R1, W1, W2, v_bias, h1_bias_up, h1_bias_down, h2_bias, v_out_prob1, v_out_prob2, v_out_prob3, h_top, h_top_prob)

def zad4():

    vae = VAE()
    vae.create_session()
    vae.init_model()
    vae.init_session()
    vae.train()
    vae.visualize()

def all():

    #zad 1
    rbm = RBM()
    rbm.init_model()
    rbm.init_session()
    w1, v_bias, h1_bias_up, v_out_prob1, h_top = rbm.train()
    rbm.visualize(w1, v_bias, v_out_prob1, h_top)

    # zad 2
    dbn = DBN()
    dbn.init_model(w1, v_bias, h1_bias_up)
    dbn.init_session()
    w1, w2, v_bias, h1_bias_down, h2_bias, v_out_prob2, h_top = dbn.train()
    dbn.visualize(w1, w2, v_bias, h1_bias_down, h2_bias, v_out_prob2, h_top)

    # zad 3
    dbnFT = DBN_Fine_Tune()
    dbnFT.init_model(w1, w2, v_bias, h1_bias_up, h1_bias_down, h2_bias)
    dbnFT.init_session()
    R1, W1, W2, v_bias, h1_bias_up, h1_bias_down, h2_bias, v_out_prob3, h_top, h_top_prob = dbnFT.train()
    dbnFT.visualize(R1, W1, W2, v_bias, h1_bias_up, h1_bias_down, h2_bias, v_out_prob1, v_out_prob2, v_out_prob3, h_top,
                    h_top_prob)

    # zad 4
    vae = VAE()
    vae.create_session()
    vae.init_model()
    vae.init_session()
    vae.train()
    vae.visualize()

def main():

    # all()
    # zad1()
    # zad2()
    # zad3()
    zad4()


if __name__ == "__main__":

    main()