include("utils/util.jl")
include("utils/newtonUpdate.jl")
include("utils/rationalSOS.jl")

X = nothing
eig_tol = 1e-6
tol = 1e-50

SOS_time = 0
Newton_time = 0
function solve(n, X, f, Q, mono,newton)
    global SOS_time, Newton_time
    Q = round.(Q, digits=2)
    rQ = symplify_rational.(Q)

    model = nothing
    if !newton
        SOS_time+=@elapsed model = rational_SOS(f, 1, 0, mono, rQ)
    else
        Newton_time+=@elapsed update_Q = newton_refine_update(f, mono, Q, eig_tol, 90, tol, X)
        Q_res = symplify_rational.(update_Q*update_Q')
        SOS_time+=@elapsed model = rational_SOS(f, 1, 0, mono, Q_res)
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


#H5
# third
## Multiplier-2
P0 = 37893*X[1]^2//1000 - 12089*X[1]*X[2]//1000 - 3689*X[1]//250 + 87*X[2]^2//2 + 1254*X[2]//125 + 7291//250
M0 = [1, X[2], X[1]]
Q0 = [29.16370848906521 5.0101937435432635 -7.367609057479814;
 5.0101937435432635 43.49770301575635 -6.039297168370388;
 -7.367609057479814 -6.039297168370388 37.88640150882121;
]
rational_SOS(P0, 1, 0, M0, Q0)
solve(n, X, P0, Q0, M0, false)
## Total Decomposition
P1 = 61103*X[1]^4//1250 + 45429*X[1]^3*X[2]//1250 - 143253*X[1]^3//20000 + 989403*X[1]^2*X[2]^2//10000 + 405239*X[1]^2*X[2]//100000 + 120403*X[1]^2//1000 + 41167*X[1]*X[2]^3//2500 - 445477*X[1]*X[2]^2//10000 - 200121*X[1]*X[2]//10000 + 79833*X[1]//1000 + 87767*X[2]^4//1250 + 110833*X[2]^3//12500 + 697061*X[2]^2//10000 - 348977*X[2]//10000 + 29879//250
M1 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q1 = [119.51644660356041 -17.507921028962976 39.88540509513845 0.17978260995653833 3.7966794860935704 11.939950029897306;
 -17.507921028962976 69.24146640629662 -13.838661084340574 4.470893574842128 -9.161966115735979 -0.9353207846689708;
 39.88540509513845 -13.838661084340574 96.48039585428317 -13.133170831469728 2.9828428170431978 -3.5926697565655115;
 0.17978260995653833 4.470893574842128 -13.133170831469728 70.21510535560203 8.23018047609141 9.83783186282073;
 3.7966794860935704 -9.161966115735979 2.9828428170431978 8.23018047609141 79.22059968647565 18.179785432895027;
 11.939950029897306 -0.9353207846689708 -3.5926697565655115 9.83783186282073 18.179785432895027 48.88376058603031;
]
solve(n, X, P1, Q1, M1, false)
## Multiplier-1
P2 = 37063*X[1]^2//1000 - 1563*X[1]*X[2]//625 + 4457*X[1]//1000 + 1343*X[2]^2//40 + 23183*X[2]//10000 + 44581//1000
M2 = [1, X[2], X[1]]
Q2 = [44.58054186950828 1.164901862297848 2.225842372574633;
 1.164901862297848 33.57526736939584 -1.2575864848294367;
 2.225842372574633 -1.2575864848294367 37.062901197096906;
]
solve(n, X, P2, Q2, M2, false)
# second
## Multiplier-2
P3 = 5081*X[1]^2//1000 - 5231*X[1]*X[2]//10000 + 7867*X[1]//100000 + 571*X[2]^2//100 + 577*X[2]//500 + 1401//250
M3 = [1, X[2], X[1]]
Q3 = [5.60353169832065 0.57581194114354 0.03720157956814577;
 0.57581194114354 5.709825787233064 -0.26239642440274513;
 0.03720157956814577 -0.26239642440274513 5.080878020137263;
]
solve(n, X, P3, Q3, M3, false)
## Total Decomposition
P4 = 14493*X[1]^4//2500 - 837*X[1]^3*X[2]//200 + 4343*X[1]^3//2500 + 10357*X[1]^2*X[2]^2//1000 - 2863*X[1]^2*X[2]//1250 + 6063*X[1]^2//500 + 6421*X[1]*X[2]^3//5000 + 28001*X[1]*X[2]^2//100000 + 1453*X[1]*X[2]//250 - 86373*X[1]//10000 + 65857*X[2]^4//10000 - 9537*X[2]^3//2500 + 5771*X[2]^2//500 - 2349*X[2]//500 + 14589//1000
M4 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q4 = [14.58884252220181 -2.3507190899622454 -4.316549383935998 1.4278567393436907 1.4953017127356165 0.777393347838418;
 -2.3507190899622454 8.681835212210219 1.4199866060253965 -1.909235900311388 0.522311470857499 -0.03888223133428113;
 -4.316549383935998 1.4199866060253965 10.57915629794573 -0.3835119599719048 -1.1044306076312225 0.867476865324688;
 1.4278567393436907 -1.909235900311388 -0.3835119599719048 6.585560232957719 0.6414775865677405 0.029166474890968432;
 1.4953017127356165 0.522311470857499 -1.1044306076312225 0.6414775865677405 10.294541715846696 -2.0883801845344614;
 0.777393347838418 -0.03888223133428113 0.867476865324688 0.029166474890968432 -2.0883801845344614 5.796362626868253;
]
solve(n, X, P4, Q4, M4, false)
## Multiplier-1
P5 = 3241*X[1]^2//500 + 2927*X[1]*X[2]//5000 - 4087*X[1]//50000 + 1077*X[2]^2//200 + 6087*X[2]//10000 + 5073//1000
M5 = [1, X[2], X[1]]
Q5 = [5.0731658846966665 0.3045827929571011 -0.042820192183598976;
 0.3045827929571011 5.384997264590656 0.2901996022290214;
 -0.042820192183598976 0.2901996022290214 6.481799767183013;
]
solve(n, X, P5, Q5, M5, false)
# seventh
## Multiplier-2
P6 = 68*X[1]^2//125 - 223*X[1]*X[2]//2500 + 89*X[1]//2000 + 14*X[2]^2//25 + 317*X[2]//1000 + 109//250
M6 = [1, X[2], X[1]]
Q6 = [0.4361275430479985 0.15869209912077753 0.022141899536948742;
 0.15869209912077753 0.5593291952247916 -0.04471293975442797;
 0.022141899536948742 -0.04471293975442797 0.5439073088572359;
]
solve(n, X, P6, Q6, M6, false)
## Total Decomposition
P7 = 2083*X[1]^4//5000 - 1223*X[1]^3*X[2]//10000 - 511*X[1]^3//500 + 9793*X[1]^2*X[2]^2//10000 - 341*X[1]^2*X[2]//400 + 1199*X[1]^2//1000 - 8793*X[1]*X[2]^3//100000 - 133*X[1]*X[2]^2//125 - 889*X[1]*X[2]//2000 - 4817*X[1]//10000 + 2797*X[2]^4//5000 - 9279*X[2]^3//10000 + 133*X[2]^2//100 - 2951*X[2]//5000 + 2361//1000
M7 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q7 = [2.3609221862247174 -0.2958527218412896 -0.24198345247572484 0.0029801269053971614 -0.4956947752165051 -0.1144846939866308;
 -0.2958527218412896 1.3250691031511963 0.2735629663579804 -0.4635616372209386 -0.2682853088321963 -0.16314096917979698;
 -0.24198345247572484 0.2735629663579804 1.4256990201779558 -0.26241551410576786 -0.2621067080537374 -0.511141107115001;
 0.0029801269053971614 -0.4635616372209386 -0.26241551410576786 0.5593291952247899 -0.044712939754430955 0.09692036495610512;
 -0.4956947752165051 -0.2682853088321963 -0.2621067080537374 -0.044712939754430955 0.7849525886080605 -0.06107291092194126;
 -0.1144846939866308 -0.16314096917979698 -0.511141107115001 0.09692036495610512 -0.06107291092194126 0.41713729374238523;
]
solve(n, X, P7, Q7, M7, false)
## Multiplier-1
P8 = 417*X[1]^2//1000 - 61*X[1]*X[2]//500 + 229*X[1]//1000 + 87*X[2]^2//200 - 679*X[2]//100000 + 339//1000
M8 = [1, X[2], X[1]]
Q8 = [0.33921637194911125 -0.0033695224254003085 0.1145648334801926;
 -0.0033695224254003085 0.43488600967525154 -0.061072910921942385;
 0.1145648334801926 -0.061072910921942385 0.4171372937423707;
]
solve(n, X, P8, Q8, M8, false)
# eighth
## Multiplier-2
P9 = 38309*X[1]^2//1000 - 38273*X[1]*X[2]//10000 + 3237*X[1]//1250 + 21959*X[2]^2//500 + 12373*X[2]//500 + 8097//125
M9 = [1, X[2], X[1]]
Q9 = [64.77606536588557 12.403623399868266 1.2837068991602454;
 12.403623399868266 43.93025001739367 -1.9090417820257344;
 1.2837068991602454 -1.9090417820257344 38.30782007578548;
]
solve(n, X, P9, Q9, M9, false)
## Total Decomposition
P10 = 5554*X[1]^4//125 - 47069*X[1]^3*X[2]//10000 - 10139*X[1]^3//200 + 76197*X[1]^2*X[2]^2//1000 - 5887*X[1]^2*X[2]//100 + 85291*X[1]^2//1000 - 38427*X[1]*X[2]^3//10000 - 6911*X[1]*X[2]^2//125 + 32353*X[1]*X[2]//1000 + 931*X[1]//40 + 43921*X[2]^4//1000 - 49837*X[2]^3//1000 + 5481*X[2]^2//50 + 5043*X[2]//100 + 11409//200
M10 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q10 = [57.044898034281914 25.187392012566672 11.6375945937492 -0.7274286808549291 -6.166623737349663 -2.332970358741874;
 25.187392012566672 111.08257355556759 22.296059986854377 -24.937088853071625 -16.9661816250046 -14.258834333334827;
 11.6375945937492 22.296059986854377 89.95838553460273 -10.695311460641816 -15.206947208335233 -25.309779881391055;
 -0.7274286808549291 -24.937088853071625 -10.695311460641816 43.93025001739369 -1.9090417820257335 5.740251438810595;
 -6.166623737349663 -16.9661816250046 -15.206947208335233 -1.9090417820257335 64.69857745372096 -2.3253798079213093;
 -2.332970358741874 -14.258834333334827 -25.309779881391055 5.740251438810595 -2.3253798079213093 44.414345540628666;
]
solve(n, X, P10, Q10, M10, false)
## Multiplier-1
P11 = 22201*X[1]^2//500 - 46149*X[1]*X[2]//10000 + 12419*X[1]//500 + 37871*X[2]^2//1000 - 17091*X[2]//10000 + 56971//1000
M11 = [1, X[2], X[1]]
Q11 = [56.97098465851377 -0.8572803513302583 12.442413563413263;
 -0.8572803513302583 37.87126025555667 -2.3253798079213137;
 12.442413563413263 -2.3253798079213137 44.4143455406287;
]
solve(n, X, P11, Q11, M11, false)
# first
## Total Decomposition
P12 = 3897*X[1]^4//250 - 28457*X[1]^3*X[2]//100000 - 857*X[1]^3//50 + 31773*X[1]^2*X[2]^2//1000 - 9051*X[1]^2*X[2]//2500 + 5971*X[1]^2//200 - 13179*X[1]*X[2]^3//50000 - 8987*X[1]*X[2]^2//500 + 11159*X[1]*X[2]//2000 + 2816*X[1]//125 + 16191*X[2]^4//1000 - 8709*X[2]^3//2000 + 30569*X[2]^2//1000 + 11387*X[2]//1250 + 10167//250
M12 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q12 = [40.6680034483869 4.57486924796815 11.265836223420928 1.6369613981772788 -0.009543803249769165 -1.1077352932041038;
 4.57486924796815 27.321622656455606 2.79289197024933 -2.16163930053627 -5.884776420880297 -0.9319312403574933;
 11.265836223420928 2.79289197024933 32.05268895343695 -3.1143378056643183 -0.8995510072937845 -8.569084763254619;
 1.6369613981772788 -2.16163930053627 -3.1143378056643183 16.190019206950225 -0.13097040220279274 2.432481311781883;
 -0.009543803249769165 -5.884776420880297 -0.8995510072937845 -0.13097040220279274 26.91205376811894 -0.13097040220279274;
 -1.1077352932041038 -0.9319312403574933 -8.569084763254619 2.432481311781883 -0.13097040220279274 15.586997184746235;
]
solve(n, X, P12, Q12, M12, false)
## Multiplier-1
P13 = 15591*X[1]^2//1000 - 171*X[1]*X[2]//625 + 9767*X[1]//1250 + 1619*X[2]^2//100 + 21543*X[2]//10000 + 421//25
M13 = [1, X[2], X[1]]
Q13 = [16.840392350064665 1.076364589101036 3.9005131703428444;
 1.076364589101036 16.19001920695023 -0.13097040220279207;
 3.9005131703428444 -0.13097040220279207 15.586997184746231;
]
solve(n, X, P13, Q13, M13, false)
# fifth
## Multiplier-2
P14 = 1261*X[1]^2//250 - 7*X[1]*X[2]//5 - 1023*X[1]//1000 + 1457*X[2]^2//250 + 7783*X[2]//10000 + 4373//1000
M14 = [1, X[2], X[1]]
Q14 = [4.3733549399663865 0.3910517693817308 -0.5096769302381413;
 0.3910517693817308 5.82835751721275 -0.7020379811090413;
 -0.5096769302381413 -0.7020379811090413 5.043856016562835;
]
solve(n, X, P14, Q14, M14, false)
## Total Decomposition
P15 = 12529*X[1]^4//2000 + 50083*X[1]^3*X[2]//10000 - 11119*X[1]^3//10000 + 13521*X[1]^2*X[2]^2//1000 + 63389*X[1]^2*X[2]//100000 + 3309*X[1]^2//200 + 25911*X[1]*X[2]^3//10000 - 54847*X[1]*X[2]^2//10000 - 28109*X[1]*X[2]//10000 + 37733*X[1]//5000 + 43987*X[2]^4//5000 + 14611*X[2]^3//10000 + 99*X[2]^2//8 - 15147*X[2]//5000 + 15787//1000
M15 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q15 = [15.787057213958295 -1.51632352391326 3.7750270504424357 1.7504867733170868 0.04905105777410805 2.495046473140162;
 -1.51632352391326 8.870790918717733 -1.4549969830709129 0.7267334444548357 -1.3174758387942593 0.060181755731198154;
 3.7750270504424357 -1.4549969830709129 11.5573202282304 -1.424467335234441 0.2558356042215416 -0.5547893473233658;
 1.7504867733170868 0.7267334444548357 -1.424467335234441 8.794590773914637 1.2987109702932926 1.51093365575167;
 0.04905105777410805 -1.3174758387942593 0.2558356042215416 1.2987109702932926 10.500664100717923 2.501926645513819;
 2.495046473140162 0.060181755731198154 -0.5547893473233658 1.51093365575167 2.501926645513819 6.263328044850357;
]
solve(n, X, P15, Q15, M15, false)
## Multiplier-1
P16 = 607*X[1]^2//125 + 41*X[1]*X[2]//2500 + 232*X[1]//125 + 4011*X[2]^2//1000 - 4503*X[2]//5000 + 2297//500
M16 = [1, X[2], X[1]]
Q16 = [4.594346104937167 -0.4491693621825416 0.92761637735049;
 -0.4491693621825416 4.011312881518845 0.00843038542668403;
 0.92761637735049 0.00843038542668403 4.855516056270657;
]
solve(n, X, P16, Q16, M16, false)
# fourth
## Multiplier-2
P17 = 37409*X[1]^2//1000 - 10387*X[1]*X[2]//10000 + 2396*X[1]//625 + 1024*X[2]^2//25 + 14063*X[2]//1000 + 10653//250
M17 = [1, X[2], X[1]]
Q17 = [42.61218097886428 7.022252249091163 1.9381364823598595;
 7.022252249091163 40.957498504521205 -0.5178401704259913;
 1.9381364823598595 -0.5178401704259913 37.41054637731726;
]
solve(n, X, P17, Q17, M17, false)
## Total Decomposition
P18 = 234679*X[1]^4//5000 - 56361*X[1]^3*X[2]//2500 - 366879*X[1]^3//50000 + 803887*X[1]^2*X[2]^2//10000 - 42497*X[1]^2*X[2]//1250 + 72977*X[1]^2//500 + 115607*X[1]*X[2]^3//20000 + 327323*X[1]*X[2]^2//10000 + 396379*X[1]*X[2]//10000 - 2129*X[1]//20 + 222639*X[2]^4//5000 - 341869*X[2]^3//10000 + 801709*X[2]^2//10000 - 227233*X[2]//10000 + 16716//125
M18 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q18 = [133.72810182679015 -11.372779306566308 -53.23189517735465 4.673820652283802 15.638795368263171 20.714756888723283;
 -11.372779306566308 70.8212557349771 4.192017060983026 -17.06110076688387 12.658993768762208 -1.6335659774335813;
 -53.23189517735465 4.192017060983026 104.50751725259936 3.6939168482583633 -15.377834700717584 -3.6656256595596095;
 4.673820652283802 -17.06110076688387 3.6939168482583633 44.5085579085732 2.8970974768609823 -0.5970496901252771;
 15.638795368263171 12.658993768762208 -15.377834700717584 2.8970974768609823 81.60647769938673 -11.25869158604258;
 20.714756888723283 -1.6335659774335813 -3.6656256595596095 -0.5970496901252771 -11.25869158604258 46.92501216415272;
]
solve(n, X, P18, Q18, M18, false)
## Multiplier-1
P19 = 51401*X[1]^2//1000 + 17021*X[1]*X[2]//100000 + 9343*X[1]//2500 + 45521*X[2]^2//1000 + 5449*X[2]//1000 + 38951//1000
M19 = [1, X[2], X[1]]
Q19 = [38.950623460572224 2.7101538937212 1.8723570943894963;
 2.7101538937212 45.51951449285362 0.07893451573774643;
 1.8723570943894963 0.07893451573774643 51.40107138941676;
]
solve(n, X, P19, Q19, M19, false)
# sixth
## Multiplier-2
P20 = 397*X[1]^2//1000 - 39*X[1]*X[2]//2000 + 479*X[1]//10000 + 227*X[2]^2//500 + 23*X[2]//125 + 203//500
M20 = [1, X[2], X[1]]
Q20 = [0.4057762941958831 0.0915160020139599 0.023828176341197382;
 0.0915160020139599 0.45397683970109687 -0.009579673477797257;
 0.023828176341197382 -0.009579673477797257 0.3972970082849449;
]
solve(n, X, P20, Q20, M20, false)
## Total Decomposition
P21 = 499*X[1]^4//1000 - 151*X[1]^3*X[2]//2500 - 201*X[1]^3//500 + 163*X[1]^2*X[2]^2//200 - 623*X[1]^2*X[2]//1000 + 173*X[1]^2//250 - 93*X[1]*X[2]^3//5000 - 337*X[1]*X[2]^2//1000 - 127*X[1]*X[2]//1000 - 77*X[1]//200 + 453*X[2]^4//1000 - 623*X[2]^3//1000 + 477*X[2]^2//1000 - 287*X[2]//500 + 909//1000
M21 = [1, X[2], X[1], X[2]^2, X[1]*X[2], X[1]^2]
Q21 = [0.9085739280276257 -0.2870916074713346 -0.19300717730773478 -0.15017718824535997 -0.1545020125706784 -0.015121312339796354;
 -0.2870916074713346 0.7763885250011618 0.09122792746995134 -0.31138843713451514 -0.0943249330291656 -0.14492490539517489;
 -0.19300717730773478 0.09122792746995134 0.7229405123511288 -0.07391466765604404 -0.16692326686254977 -0.2007226820179002;
 -0.15017718824535997 -0.31138843713451514 -0.07391466765604404 0.4539768397010928 -0.00957967347779611 0.07626155752367618;
 -0.1545020125706784 -0.0943249330291656 -0.16692326686254977 -0.00957967347779611 0.6629172970260278 -0.03001960174183629;
 -0.015121312339796354 -0.14492490539517489 -0.2007226820179002 0.07626155752367618 -0.03001960174183629 0.4988201620047996;
]
solve(n, X, P21, Q21, M21, false)
## Multiplier-1
P22 = 499*X[1]^2//1000 - 301*X[1]*X[2]//5000 + 61*X[1]//625 + 209*X[2]^2//500 + 27*X[2]//1250 + 54//125
M22 = [1, X[2], X[1]]
Q22 = [0.43230747242519335 0.010733316113748106 0.048687393029795266;
 0.010733316113748106 0.41814340454834314 -0.03001960174183667;
 0.048687393029795266 -0.03001960174183667 0.4988201620047993;
]
solve(n, X, P22, Q22, M22, false)

@show SOS_time Newton_time