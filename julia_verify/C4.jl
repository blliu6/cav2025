include("utils/util.jl")
include("utils/newtonUpdate.jl")
include("utils/rationalSOS.jl")

X = nothing
eig_tol = 1e-4
tol = 1e-50


function solve(n, X, f, Q, mono,newton)
    
    Q = round.(Q, digits=2)
    rQ = symplify_rational.(Q)

    model = nothing
    if !newton
        model = rational_SOS(f, 1, 0, mono, rQ)
    else
        update_Q = newton_refine_update(f, mono, Q, eig_tol, 90, tol, X)
        Q_res = symplify_rational.(update_Q*update_Q')
        model = rational_SOS(f, 1, 0, mono, Q_res)
    end
    
    SOSrational_time, coeffs, terms = model[2:end]
    
    @show expand(f-(coeffs' * terms.^2))

    data = Dict(
        "n" => n,
        "expression" => f,
        "coeff" => join([string(c) for c in coeffs], ","),
        "term" => join([string(t) for t in terms], ","),
        "SOSrational_time" => SOSrational_time,
        "SOS_expression" => string(coeffs' * terms.^2)
    )

    # save("src/output/data.json", data)    
end

n = 2
@Symbolics.variables X[1:n]

#C4
# init
## Multiplier-2
P0 = 14043*X[1]^2//1000 - 86679*X[1]*X[2]//100000 + 2077*X[1]//2000 + 12047*X[2]^2//1000 + 13349*X[2]//5000 + 5297//500
M0 = [1, X[2], X[1]]
Q0 = [10.594074879498054 1.3366910653172779 0.5216243396719706;
 1.3366910653172779 12.047393534382357 -0.43678299519060537;
 0.5216243396719706 -0.43678299519060537 14.043772185945617;
]
solve(n, X, P0, Q0, M0, true)
## Total Decomposition
P1 = 10623*X[1]^4//500 - 42853*X[1]^3*X[2]//100000 + 6297*X[1]^3//10000 + 30221*X[1]^2*X[2]^2//1000 - 3746*X[1]^2*X[2]//125 + 16933*X[1]^2//500 - 86121*X[1]*X[2]^3//100000 + 23777*X[1]*X[2]^2//250000 + 5501*X[1]*X[2]//500 - 17651*X[1]//500 + 2407*X[2]^4//200 - 29003*X[2]^3//1000 + 37173*X[2]^2//1000 - 7389*X[2]//100 + 17169//250
M1 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q1 = [68.67565176391099 -36.98145410383155 -17.628171816940633 -4.894548132145514 -2.0995588507584233 3.6286071435498655;
 -36.98145410383155 46.96817438024146 7.630540909520901 -14.498746283890284 -1.544607638817718 -8.967560436628043;
 -17.628171816940633 7.630540909520901 26.578275453548834 1.5886447314216863 -6.034947193911623 0.32432548755463864;
 -4.894548132145514 -14.498746283890284 1.5886447314216863 12.047393534382305 -0.4367829951905555 3.70081630429404;
 -2.0995588507584233 -1.544607638817718 -6.034947193911623 -0.4367829951905555 22.833571879329426 -0.21551507244026227;
 3.6286071435498655 -8.967560436628043 0.32432548755463864 3.70081630429404 -0.21551507244026227 21.249743560237615;
]
solve(n, X, P1, Q1, M1, true)
## Multiplier-1
P2 = 85*X[1]^2//4 - 21143*X[1]*X[2]//50000 - 889*X[1]//400 + 16191*X[2]^2//1000 + 16939*X[2]//10000 + 4707//250
M2 = [1, X[2], X[1]]
Q2 = [18.828157894008974 0.8465213630547227 -1.1066016711050028;
 0.8465213630547227 16.1914323019734 -0.21551507244027807;
 -1.1066016711050028 -0.21551507244027807 21.24974356023801;
]
solve(n, X, P2, Q2, M2, true)
# unsafe
## Multiplier-2
P3 = 8399*X[1]^2//2500 - 29037*X[1]*X[2]//10000 - 45087*X[1]//10000 + 39723*X[2]^2//10000 + 1136*X[2]//625 + 4369//250
M3 = [1, X[2], X[1]]
Q3 = [17.476457870143047 0.9109464888055456 -2.260335885470591;
 0.9109464888055456 3.9726901951324183 -1.4540960040946829;
 -2.260335885470591 -1.4540960040946829 3.362506921232156;
]
solve(n, X, P3, Q3, M3, true)
## Total Decomposition
P4 = 29047*X[1]^4//10000 - 31017*X[1]^3*X[2]//10000 - 19823*X[1]^3//2500 + 15611*X[1]^2*X[2]^2//2000 - 12193*X[1]^2*X[2]//2500 + 7299*X[1]^2//500 - 1817*X[1]*X[2]^3//625 - 30997*X[1]*X[2]^2//5000 - 2351*X[1]*X[2]//10000 - 726*X[1]//125 + 39721*X[2]^4//10000 - 5529*X[2]^3//1000 + 8773*X[2]^2//1000 - 4143*X[2]//1000 + 2051//500
M4 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q4 = [4.102132320577922 -2.0730038061257856 -2.902627816098071 0.457419657914861 -0.14906846808335705 0.9832707750643197;
 -2.0730038061257856 7.8552227741250675 0.0279658236784108 -2.7659691808658917 -1.5081410426930442 -0.19456578425395188;
 -2.902627816098071 0.0279658236784108 12.628416984433995 -1.5840547833304914 -2.2425043892683814 -3.960415374849352;
 0.457419657914861 -2.7659691808658917 -1.5840547833304914 3.9726901951324347 -1.4540960040945847 0.7986991920205719;
 -0.14906846808335705 -1.5081410426930442 -2.2425043892683814 -1.4540960040945847 6.201439829912315 -1.5513349256741182;
 0.9832707750643197 -0.19456578425395188 -3.960415374849352 0.7986991920205719 -1.5513349256741182 2.901776354519017;
]
solve(n, X, P4, Q4, M4, true)
## Multiplier-1
P5 = 5809*X[1]^2//2000 - 15541*X[1]*X[2]//5000 + 18283*X[1]//5000 + 44373*X[2]^2//10000 - 2099*X[2]//500 + 6771//500
M5 = [1, X[2], X[1]]
Q5 = [13.541646986563325 -2.097099853101808 1.8231763155891414;
 -2.097099853101808 4.436331292722108 -1.5513349256739282;
 1.8231763155891414 -1.5513349256739282 2.9017763545195345;
]
solve(n, X, P5, Q5, M5, true)
# Lie
## Multiplier-2
P6 = 15459*X[1]^4//10000 + 22341*X[1]^3*X[2]//5000 - 3761*X[1]^3//500 + 64731*X[1]^2*X[2]^2//10000 - 2334*X[1]^2*X[2]//125 + 6031*X[1]^2//250 + 85061*X[1]*X[2]^3//10000 - 133*X[1]*X[2]^2//4 + 18173*X[1]*X[2]//250 - 33063*X[1]//500 + 11057*X[2]^4//1250 - 36467*X[2]^3//1000 + 10774*X[2]^2//125 - 3087*X[2]//25 + 36493//500
M6 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q6 = [72.98575332864505 -61.74421783241358 -33.075132482140596 9.352454471979325 9.045394028313378 0.47406303835393393;
 -61.74421783241358 67.51610985604746 27.305786483079533 -18.24586136694469 -13.215023175396334 -0.18296724026831937;
 -33.075132482140596 27.305786483079533 23.263154712859222 -3.4109266094846356 -9.176352223160247 -3.773888126758965;
 9.352454471979325 -18.24586136694469 -3.4109266094846356 8.851705891541355 4.2551866667397915 -0.20446289207730098;
 9.045394028313378 -13.215023175396334 -9.176352223160247 4.2551866667397915 6.880085572423342 2.233506088951028;
 0.47406303835393393 -0.18296724026831937 -3.773888126758965 -0.20446289207730098 2.233506088951028 1.5430051554867188;
]
solve(n, X, P6, Q6, M6, true)
## Total Decomposition
P7 = 7523*X[1]^6//2500 + 68081*X[1]^5*X[2]//10000 - 13119*X[1]^5//1000 + 1566*X[1]^4*X[2]^2//125 - 24969*X[1]^4*X[2]//1000 + 533*X[1]^4//40 + 10377*X[1]^3*X[2]^3//1000 - 42313*X[1]^3*X[2]^2//1000 + 4864*X[1]^3*X[2]//125 + 11497*X[1]^3//5000 + 14221*X[1]^2*X[2]^4//1000 - 14771*X[1]^2*X[2]^3//250 + 10693*X[1]^2*X[2]^2//125 - 25119*X[1]^2*X[2]//500 + 3007*X[1]^2//250 + 1063*X[1]*X[2]^5//125 - 33233*X[1]*X[2]^4//1000 + 3247*X[1]*X[2]^3//50 - 55327*X[1]*X[2]^2//1000 + 35657*X[1]*X[2]//1000 - 4681*X[1]//100 + 88567*X[2]^6//10000 - 4561*X[2]^5//125 + 29443*X[2]^4//500 - 6667*X[2]^3//125 + 5584*X[2]^2//125 - 1273*X[2]//25 + 11657//250
M7 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2, X[2]^3, X[1]*X[2]^2, X[1]^2*X[2], X[1]^3]
Q7 = [46.62804230935786 -25.44481646488333 -23.419191041653576 -2.883821746548802 4.7475642104053035 -9.814776001998535 1.727904017796169 1.2015555140294325 2.4641810768870736 5.101841443426479;
 -25.44481646488333 50.43787674087979 13.06798336834355 -28.40836371050573 -21.801280168491168 -5.796553629902836 4.331275701063291 7.10079604496183 4.842007958381883 2.2164757410065334;
 -23.419191041653576 13.06798336834355 31.62528598128617 -7.053075519772084 -21.77868146463037 -3.920037568403403 3.6262815451951895 4.971249703685117 3.7925199881152416 -2.653412899986089;
 -2.883821746548802 -28.40836371050573 -7.053075519772084 50.255041559117174 21.74330779947428 10.106035815247044 -18.24586136562699 -11.350322591283199 -8.991992075200978 -2.4959449990700286;
 4.7475642104053035 -21.801280168491168 -21.77868146463037 21.74330779947428 45.703106267920894 13.421767589417259 -5.275627192980104 -18.834154248759404 -15.114994173711708 -2.001707327185855;
 -9.814776001998535 -5.796553629902836 -3.920037568403403 10.106035815247044 13.421767589417259 18.64391256484581 -1.7144019522532254 -3.544646981910187 -10.492819285141366 -6.571023213651603;
 1.727904017796169 4.331275701063291 3.6262815451951895 -18.24586136562699 -5.275627192980104 -1.7144019522532254 8.851705892764585 4.2551866670176635 1.6736662045374031 -0.2408727237628431;
 1.2015555140294325 7.10079604496183 4.971249703685117 -11.350322591283199 -18.834154248759404 -3.544646981910187 4.2551866670176635 10.8741892107217 5.435506355585272 0.5541654260685539;
 2.4641810768870736 4.842007958381883 3.7925199881152416 -8.991992075200978 -15.114994173711708 -10.492819285141366 1.6736662045374031 5.435506355585272 11.43557912687442 3.4127063023149673;
 5.101841443426479 2.2164757410065334 -2.653412899986089 -2.4959449990700286 -2.001707327185855 -6.571023213651603 -0.2408727237628431 0.5541654260685539 3.4127063023149673 3.0121214457589014;
]
solve(n, X, P7, Q7, M7, true)
## Multiplier-1
P8 = 30117*X[1]^4//10000 + 8513*X[1]^3*X[2]//1250 - 13117*X[1]^3//1000 + 5493*X[1]^2*X[2]^2//500 - 21061*X[1]^2*X[2]//500 + 47209*X[1]^2//1000 + 59099*X[1]*X[2]^3//10000 - 4291*X[1]*X[2]^2//125 + 10589*X[1]*X[2]//100 - 2598*X[1]//25 + 38703*X[2]^4//5000 - 8623*X[2]^3//250 + 17133*X[2]^2//200 - 683*X[2]//5 + 92297//1000
M8 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q8 = [92.29736511508602 -68.28905934364603 -51.92093541396324 7.955323628430268 19.599330885359777 1.033425289656706;
 -68.28905934364603 69.7619711095025 33.33148737286837 -17.26053083934349 -15.772115799887997 -0.8349127268152746;
 -51.92093541396324 33.33148737286837 45.13191255037831 -1.4104410920284423 -20.245176834079878 -6.571023217299931;
 7.955323628430268 -17.26053083934349 -1.4104410920284423 7.7503618363085955 2.9611275414334477 -0.29499433460719193;
 19.599330885359777 -15.772115799887997 -20.245176834079878 2.9611275414334477 11.590893500198343 3.4127063031450535;
 1.033425289656706 -0.8349127268152746 -6.571023217299931 -0.29499433460719193 3.4127063031450535 3.0121214553039777;
]
solve(n, X, P8, Q8, M8, true)