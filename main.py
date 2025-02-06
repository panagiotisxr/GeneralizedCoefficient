from math import sin, cos, tan, pi, radians, sqrt, asin, atan
from js import document




class ElementWrapper:
    def __init__(self, id):
        self.element = document.getElementById(id)
    def write(self, text):
        self.element.innerHTML = str(text)

def Element(id):
    return ElementWrapper(id)

def write_func(id, text):
    document.getElementById(id).innerHTML = str(text)

# The rest of your code remains unchanged except that we now call write_func instead of write

def run_all():
    results_python()

def my_function(*args, **kwargs):
    run_all()

def data():
    σv_prime = float(Element('σv_prime').element.value)
    H = float(Element('H').element.value)
    c = float(Element('c').element.value)
    φ_d = float(Element('φ_d').element.value)
    #γ = float(Element('γ').element.value)
    ah = float(Element('ah').element.value)
    av = float(Element('av').element.value)
    Es = float(Element('Es').element.value)
    ν = float(Element('ν').element.value)
    Z = float(Element('Z').element.value)
    Inte_p = int(Element('Inte_p').element.value)

    if Z > H:
        Z = H
        print("Error Z couldn't be greater than H")

    return (σv_prime, H, c, φ_d, ah, av, Es, ν, Z, Inte_p)

def fundamentals():
    σv_prime, H, c, φ_d, ah, av, Es, ν, Z, Inte_p = data()
    ah_av = ah/(1-av)
    φ_r = radians(φ_d)
    Rankine_A = (1 - sin(φ_r))/(1 + sin(φ_r))
    Rankine_P = 1/Rankine_A
    Jaky = 1 - sin(φ_r)
    ζ = complex(-0.5, sqrt(3)/2)
    return (ah_av, φ_r, Rankine_A, Rankine_P, Jaky, ζ)

def at_rest(x, y):
    σv_prime, H, c, φ_d, ah, av, Es, ν, Z, Inte_p = data()
    ah_av, φ_r, Rankine_A, Rankine_P, Jaky, ζ = fundamentals()
    z_a = Z 
    mA = x
    ξA = (mA - 1)/(mA + 1) - 1 
    A0_A = Rankine_A * (1 - ξA*sin(φ_r) + ah_av*tan(φ_r)*(2 + ξA*Jaky))
    if A0_A < 1:
        λ = 1
    else:
        λ = 0
    two_λ_1 = 2*λ - 1
    B1_A = (two_λ_1 * 2*c/((1-av)*σv_prime)) * tan(pi/4 - φ_r/2)
    e1 = (1 - A0_A) / B1_A 
    e2 = (1 + A0_A) / (two_λ_1 * B1_A) + 2*c/((1-av)*σv_prime*two_λ_1*B1_A*tan(φ_r))
    α0 = 1 + (e2**2)*(tan(φ_r)**2)
    b0 = 1 - (2*two_λ_1*e1*e2 + e2**2)*(tan(φ_r)**2)
    c0 = (e1**2 + 2*two_λ_1*e1*e2)*(tan(φ_r)**2)
    d0 = -e1**2*(tan(φ_r)**2)
    D0 = b0**2 - 3*α0*c0
    D1 = 2*b0**3 - 9*α0*b0*c0 + 27*α0**2*d0
    IMSQRT = ((D1**2 - 4*(D0**3))**0.5)
    D1_sqrt = (D1 - IMSQRT) / 2
    C0 = D1_sqrt**(1/3)
    ζ = complex(-0.5, sqrt(3)/2)
    ζ_λ_C0 = (ζ**λ) * C0
    twoλ_1_3α0 = -1/(3*two_λ_1*α0)
    D0_ζλC0 = D0 / ζ_λ_C0
    bo_ζλC0 = b0 + ζ_λ_C0 + D0_ζλC0
    colV_colX = bo_ζλC0.real * twoλ_1_3α0.real
    φm = asin(colV_colX)
    φm_deg_o = φm * (180 / pi)
    cm_o = c * tan(φm) / tan(φ_r)
    Κ_ΧΕ = ((1 - two_λ_1*sin(φm))/(1 + two_λ_1*sin(φm))
             - two_λ_1*2*cm_o*tan(pi/4 - two_λ_1*φm/2)/(σv_prime*(1-av)))
    σ_ΧΕ = Κ_ΧΕ * (1-av)*σv_prime
    At_rest = Κ_ΧΕ
    
    return (At_rest, cm_o, φm_deg_o)

def active(x, y):
    σv_prime, H, c, φ_d, ah, av, Es, ν, Z, Inte_p = data()
    ah_av, φ_r, Rankine_A, Rankine_P, Jaky, ζ = fundamentals()
    z_a = Z 
    if z_a > H/2:
        z_a = H/2
    mA = x   
    ξA = (mA - 1)/(mA + 1) - 1 
    A0_A = Rankine_A * (1 - ξA*sin(φ_r) + ah_av*tan(φ_r)*(2 + ξA*Jaky))
    if A0_A < 1:
        λ = 1
    else:
        λ = 0
    two_λ_1 = 2*λ - 1
    B1_A = (two_λ_1 * 2*c/((1-av)*σv_prime)) * tan(pi/4 - φ_r/2)
    e1 = (1 - A0_A) / B1_A 
    e2 = (1 + A0_A) / (two_λ_1 * B1_A) + 2*c/((1-av)*σv_prime*two_λ_1*B1_A*tan(φ_r))
    α0 = 1 + (e2**2)*(tan(φ_r)**2)
    b0 = 1 - (2*two_λ_1*e1*e2 + e2**2)*(tan(φ_r)**2)
    c0 = (e1**2 + 2*two_λ_1*e1*e2)*(tan(φ_r)**2)
    d0 = -e1**2*(tan(φ_r)**2)
    D0 = b0**2 - 3*α0*c0
    D1 = 2*b0**3 - 9*α0*b0*c0 + 27*α0**2*d0
    IMSQRT = ((D1**2 - 4*(D0**3))**0.5)
    D1_sqrt = (D1 - IMSQRT) / 2
    C0 = D1_sqrt**(1/3)
    ζ = complex(-0.5, sqrt(3)/2)
    ζ_λ_C0 = (ζ**λ) * C0
    twoλ_1_3α0 = -1/(3*two_λ_1*α0)
    D0_ζλC0 = D0 / ζ_λ_C0
    bo_ζλC0 = b0 + ζ_λ_C0 + D0_ζλC0
    colV_colX = bo_ζλC0.real * twoλ_1_3α0.real
    φm = asin(colV_colX)
    φm_deg_a = φm * (180 / pi)
    cm_a = c * tan(φm) / tan(φ_r)
    Κ_ΧΕ = ((1 - two_λ_1*sin(φm))/(1 + two_λ_1*sin(φm))
             - two_λ_1*2*cm_a*tan(pi/4 - two_λ_1*φm/2)/(σv_prime*(1-av)))
    σ_ΧΕ = Κ_ΧΕ * (1-av)*σv_prime
    Active = Κ_ΧΕ
    At_rest, cm_o, φm_deg_o = at_rest(1, 1)
    ΔΚ_a = At_rest - Active
    ΔxM_a = (pi/4)*((1-ν**2)/Es) * (1+z_a/H)**3 * (1-z_a/H) * H * ΔΚ_a * (1-av)*σv_prime/(z_a/H)
    return (Active, cm_a, φm_deg_a, ΔxM_a)

def passive(x, y):
    σv_prime, H, c, φ_d, ah, av, Es, ν, Z, Inte_p = data()
    ah_av, φ_r, Rankine_A, Rankine_P, Jaky, ζ = fundamentals()
    At_rest, cm_o, φm_deg_o = at_rest(1, 1)
    z_a = Z 
    if z_a > H/2:
        z_a = H/2 
    mA = x  
    ξA = (mA - 1)/(mA + 1) - 1 
    A0_A = (Rankine_P**(1+ξA)) * (1 + ξA*sin(φ_r) + (2/mA - 1)*ah_av*tan(φ_r)*(2 + ξA*(1+sin(φ_r))))
    if A0_A < 1:
        λ = 1
    else:
        λ = 0
    two_λ_1 = 2*λ - 1
    B1_A = (two_λ_1*2*c/((1-av)*σv_prime)) * tan(pi/4 - φ_r/2) * (tan(pi/4+φ_r/2)/tan(pi/4-φ_r/2))**(1+ξA)
    e1 = (1 - A0_A) / B1_A 
    e2 = (1 + A0_A) / (two_λ_1*B1_A) + 2*c/((1-av)*σv_prime*two_λ_1*B1_A*tan(φ_r))    
    α0 = 1 + (e2**2)*(tan(φ_r)**2)
    b0 = 1 - (2*two_λ_1*e1*e2 + e2**2)*(tan(φ_r)**2)
    c0 = (e1**2 + 2*two_λ_1*e1*e2)*(tan(φ_r)**2)
    d0 = -e1**2*(tan(φ_r)**2)
    D0 = b0**2 - 3*α0*c0
    D1 = 2*b0**3 - 9*α0*b0*c0 + 27*α0**2*d0
    IMSQRT = ((D1**2 - 4*(D0**3))**0.5)
    D1_sqrt = (D1 - IMSQRT) / 2
    C0 = D1_sqrt**(1/3)
    ζ = complex(-0.5, sqrt(3)/2)
    ζ_λ_C0 = (ζ**λ) * C0
    twoλ_1_3α0 = -1/(3*two_λ_1*α0)
    D0_ζλC0 = D0 / ζ_λ_C0
    bo_ζλC0 = b0 + ζ_λ_C0 + D0_ζλC0
    colV_colX = bo_ζλC0.real * twoλ_1_3α0.real
    φm = asin(colV_colX)
    φm_deg_p_s = φm * (180 / pi)
    cm_p_s = c * tan(φm) / tan(φ_r)
    Κ_ΧΕ = ((1 - two_λ_1*sin(φm))/(1 + two_λ_1*sin(φm))
             - two_λ_1*2*cm_p_s*tan(pi/4 - two_λ_1*φm/2)/(σv_prime*(1-av)))
    σ_ΧΕ = Κ_ΧΕ * (1-av)*σv_prime
    Passive = Κ_ΧΕ 
    ΔΚ_p = Passive - At_rest
    ΔxM_p = (pi/4)*((1-ν**2)/Es) * (1+z_a/H)**3 * (1-z_a/H) * H * ΔΚ_p * (1-av)*σv_prime/(z_a/H)
    return (Passive, cm_p_s, φm_deg_p_s, ΔxM_p)

def int_passive(Δx_new):
    Passive, cm_p_s, φm_deg_p_s, ΔxM_p = passive(99999999999, 1)    
    σv_prime, H, c, φ_d, ah, av, Es, ν, Z, Inte_p = data()
    ah_av, φ_r, Rankine_A, Rankine_P, Jaky, ζ = fundamentals()

    Δx_int_pass = Δx_new
    if Δx_int_pass/ΔxM_p > 1:
        Δx_ΔxM_p = 0.99999    
    else:
        Δx_ΔxM_p = Δx_int_pass/ΔxM_p

    mA_p = (1 + Δx_ΔxM_p*(H/Z)**(1+Δx_ΔxM_p)) / (1 - Δx_ΔxM_p)
    ξA_p = (mA_p - 1)/(mA_p + 1) - 1
    A0_A_p = (Rankine_P**(1+ξA_p)) * (1 + ξA_p*sin(φ_r) + (2/mA_p - 1)*ah_av*tan(φ_r)*(2+ξA_p*(1+sin(φ_r))))
    if A0_A_p < 1:
        λ_p = 1
    else:
        λ_p = 0    
    two_λ_p = 2*λ_p - 1
    B1_A_p = (two_λ_p*2*c/((1-av)*σv_prime)) * tan(pi/4-φ_r/2) * (tan(pi/4+φ_r/2)/tan(pi/4-φ_r/2))**(1+ξA_p)          
    e1_p = (1 - A0_A_p)/B1_A_p
    e2_p = (1 + A0_A_p)/(two_λ_p*B1_A_p) + 2*c/((1-av)*σv_prime*two_λ_p*B1_A_p*tan(φ_r))
    α0_p = 1 + (e2_p**2)*(tan(φ_r)**2)          
    b0_p = 1 - (2*two_λ_p*e1_p*e2_p + e2_p**2)*(tan(φ_r)**2)            
    twoλ_1_3α0_p = -1/(3*two_λ_p*α0_p) 
    c0_p = (e1_p**2 + 2*two_λ_p*e1_p*e2_p)*(tan(φ_r)**2)
    d0_p = -e1_p**2*(tan(φ_r)**2)
    D0_p = b0_p**2 - 3*α0_p*c0_p
    D1_p = 2*b0_p**3 - 9*α0_p*b0_p*c0_p + 27*α0_p**2*d0_p
    IMSQRT_p = ((D1_p**2 - 4*(D0_p**3))**0.5)
    D1_sqrt_p = (D1_p - IMSQRT_p) / 2
    C0_p = D1_sqrt_p**(1/3)
    ζ_λ_C0_p = (ζ**λ_p)*C0_p
    D0_ζλC0_p = D0_p/ζ_λ_C0_p
    bo_ζλC0_p = b0_p + ζ_λ_C0_p + D0_ζλC0_p
    colV_colX_p = bo_ζλC0_p.real * twoλ_1_3α0_p.real
    φm_p = asin(colV_colX_p)
    φm_deg_p = φm_p * (180 / pi)
    cm_p = c * tan(φm_p) / tan(φ_r)
    Κ_ΧΕ_p_2 = ((1 - two_λ_p*sin(φm_p))/(1 + two_λ_p*sin(φm_p))
                - two_λ_p*2*cm_p*tan(pi/4 - two_λ_p*φm_p/2)/(σv_prime*(1-av)))
    return (Κ_ΧΕ_p_2, cm_p, φm_deg_p)

def int_active(Δx_new):
    Active, cm_a, φm_deg_a, ΔxM_a = active(99999999999, 1)  
    σv_prime, H, c, φ_d, ah, av, Es, ν, Z, Inte_p = data()
    ah_av, φ_r, Rankine_A, Rankine_P, Jaky, ζ = fundamentals()

    Δx_int_act = Δx_new
    if Δx_int_act/ΔxM_a > 1:
        Δx_ΔxM = 0.99999    
    else:
        Δx_ΔxM = Δx_int_act/ΔxM_a

    mA = (1 + Δx_ΔxM*(H/Z)**(1+Δx_ΔxM)) / (1 - Δx_ΔxM)
    ξA = (mA - 1)/(mA + 1) - 1
    A0_A = Rankine_A * (1 - ξA*sin(φ_r) + ah_av*tan(φ_r)*(2+ξA*Jaky))
    if A0_A < 1:
        λ = 1
    else:
        λ = 0
    two_λ_1 = 2*λ - 1  
    B1_A = (2*c/((1-av)*σv_prime)) * tan(pi/4-φ_r/2)
    e1 = (1 - A0_A) / B1_A
    e2 = (1 + A0_A) / (two_λ_1*B1_A) + 2*c/((1-av)*σv_prime*two_λ_1*B1_A*tan(φ_r))
    α0 = 1 + (e2**2)*(tan(φ_r)**2)
    b0 = 1 - (2*two_λ_1*e1*e2 + e2**2)*(tan(φ_r)**2)
    c0 = (e1**2 + 2*two_λ_1*e1*e2)*(tan(φ_r)**2)
    d0 = -e1**2*(tan(φ_r)**2)
    D0 = b0**2 - 3*α0*c0
    D1 = 2*b0**3 - 9*α0*b0*c0 + 27*α0**2*d0
    IMSQRT = ((D1**2 - 4*(D0**3))**0.5)
    D1_sqrt = (D1 - IMSQRT) / 2
    C0 = D1_sqrt**(1/3)   
    ζ_λ_C0 = (ζ**λ) * C0
    twoλ_1_3α0 = -1/(3*two_λ_1*α0)
    D0_ζλC0 = D0 / ζ_λ_C0
    bo_ζλC0 = b0 + ζ_λ_C0 + D0_ζλC0
    colV_colX = bo_ζλC0.real * twoλ_1_3α0.real
    φm = asin(colV_colX)
    φm_deg = φm * (180 / pi)
    cm = c * tan(φm) / tan(φ_r)
    Κ_ΧΕ = ((1 - two_λ_1*sin(φm))/(1 + two_λ_1*sin(φm))
            - two_λ_1*2*cm*tan(pi/4 - two_λ_1*φm/2)/(σv_prime*(1-av)))
    φm_deg_int_a = φm_deg
    cm_int_a = cm
    Κ_ΧΕ_int_a = Κ_ΧΕ
    return (Κ_ΧΕ_int_a, cm_int_a, φm_deg_int_a)

def run_interm(*args, **kwargs):
    σv_prime, H, c, φ_d, ah, av, Es, ν, Z, Inte_p = data()
    ah_av, φ_r, Rankine_A, Rankine_P, Jaky, ζ = fundamentals()

    dx_dmax = []
    for i in range(1, Inte_p + 1):
        dx_dmax.append(i / Inte_p)
    res_a = dx_dmax[::-1]
    if 0 < Inte_p < 100:
        res_a[0] = 0.99
    elif 100 <= Inte_p:
        res_a[0] = 0.999
    #-----------------------------intermediate active-----------------------------------------------------
    Active, cm_a, φm_deg_a, ΔxM_a = active(99999999999, 1)
    At_rest, cm_o, φm_deg_o = at_rest(1, 1)
    Δx_int_act = []
    for i in range(0, Inte_p):
        Δx_int_act.append(ΔxM_a * res_a[i])
    mA = []
    ξA = []
    A0_A = []
    B1_A = []
    e1 = []
    e2 = []
    two_λ_1_list = []
    α0_list = []
    b0_list = []
    c0_list = []
    d0_list = []
    D0_list = []
    D1_list = []
    IMSQRT_list = []
    D1_sqrt_list = []
    C0_list = []
    ζ_λ_C0_list = []
    twoλ_1_3α0_list = []
    D0_ζλC0_list = []
    bo_ζλC0_list = []
    colV_colX_list = []
    φm_list = []
    φm_deg_int = []
    cm_int = []
    Κ_ΧΕ_int = []
    for i in range(0, Inte_p):
        if Δx_int_act[i] / ΔxM_a > 1:
            Δx_ratio = 0.99999
        else:
            Δx_ratio = Δx_int_act[i] / ΔxM_a
        mA.append((1 + Δx_ratio * (H / Z) ** (1 + Δx_ratio)) / (1 - Δx_ratio))
        ξA.append((mA[i] - 1) / (mA[i] + 1) - 1)
        A0_A.append(Rankine_A * (1 - ξA[i] * sin(φ_r) + ah_av * tan(φ_r) * (2 + ξA[i] * Jaky)))
        if A0_A[i] < 1:
            λ_val = 1
        else:
            λ_val = 0
        two_λ_1_list.append(2 * λ_val - 1)
        B1_A.append((2 * c / ((1 - av) * σv_prime)) * tan(pi / 4 - φ_r / 2))
        e1.append((1 - A0_A[i]) / B1_A[i])
        e2.append((1 + A0_A[i]) / (two_λ_1_list[i] * B1_A[i])
                  + 2 * c / ((1 - av) * σv_prime * two_λ_1_list[i] * B1_A[i] * tan(φ_r)))
        α0_list.append(1 + (e2[i] ** 2) * (tan(φ_r) ** 2))
        b0_list.append(1 - (2 * two_λ_1_list[i] * e1[i] * e2[i] + e2[i] ** 2) * (tan(φ_r) ** 2))
        c0_list.append((e1[i] ** 2 + 2 * two_λ_1_list[i] * e1[i] * e2[i]) * (tan(φ_r) ** 2))
        d0_list.append(-e1[i] ** 2 * (tan(φ_r) ** 2))
        D0_list.append(b0_list[i] ** 2 - 3 * α0_list[i] * c0_list[i])
        D1_list.append(2 * b0_list[i] ** 3 - 9 * α0_list[i] * b0_list[i] * c0_list[i] + 27 * α0_list[i] ** 2 * d0_list[i])
        IMSQRT_list.append((D1_list[i] ** 2 - 4 * (D0_list[i] ** 3)) ** 0.5)
        D1_sqrt_list.append((D1_list[i] - IMSQRT_list[i]) / 2)
        C0_list.append(D1_sqrt_list[i] ** (1 / 3))
        ζ_λ_C0_list.append((ζ ** λ_val) * C0_list[i])
        twoλ_1_3α0_list.append(-1 / (3 * two_λ_1_list[i] * α0_list[i]))
        D0_ζλC0_list.append(D0_list[i] / ζ_λ_C0_list[i])
        bo_ζλC0_list.append(b0_list[i] + ζ_λ_C0_list[i] + D0_ζλC0_list[i])
        colV_colX_list.append(bo_ζλC0_list[i].real * twoλ_1_3α0_list[i].real)
        φm_list.append(asin(colV_colX_list[i]))
        φm_deg_int.append(φm_list[i] * (180 / pi))
        cm_int.append(c * tan(φm_list[i]) / tan(φ_r))
        Κ_ΧΕ_int.append((1 - two_λ_1_list[i] * sin(φm_list[i])) / (1 + two_λ_1_list[i] * sin(φm_list[i]))
                         - two_λ_1_list[i] * 2 * cm_int[i] * tan(pi / 4 - two_λ_1_list[i] * φm_list[i] / 2)
                         / (σv_prime * (1 - av)))
    #-----------------------------intermediate passive-----------------------------------------------------
    Passive, cm_p_s, φm_deg_p_s, ΔxM_p = passive(99999999999, 1)
    dx_dmax = []
    for i in range(1, Inte_p + 1):
        dx_dmax.append(i / Inte_p)
    res_p = dx_dmax[:]
    if 0 < Inte_p < 100:
        res_p[Inte_p - 1] = 0.99
    elif 100 <= Inte_p:
        res_p[Inte_p - 1] = 0.999 
    Δx_int_pass = []
    for i in range(0, Inte_p):
        Δx_int_pass.append(ΔxM_p * res_p[i])
    Δx_ΔxM_p = []
    mA_p = []
    ξA_p = []
    A0_A_p_list = []
    B1_A_p = []
    e1_p_list = []
    e2_p_list = []
    two_λ_p_list = []
    α0_p_list = []
    b0_p_list = []
    c0_p_list = []
    d0_p_list = []
    D0_p_list = []
    D1_p_list = []
    IMSQRT_p_list = []
    D1_sqrt_p_list = []
    C0_p_list = []
    ζ_λ_C0_p_list = []
    twoλ_1_3α0_p_list = []
    D0_ζλC0_p_list = []
    bo_ζλC0_p_list = []
    colV_colX_p_list = []
    φm_p_list = []
    φm_deg_p_int = []
    cm_p_int = []
    Κ_ΧΕ_p_int = []
    for i in range(0, Inte_p):
        if Δx_int_pass[i] / ΔxM_p > 1:
            Δx_ΔxM_p.append(0.99999)    
        else:
            Δx_ΔxM_p.append(Δx_int_pass[i] / ΔxM_p)
        mA_p.append((1 + Δx_ΔxM_p[i] * (H / Z) ** (1 + Δx_ΔxM_p[i])) / (1 - Δx_ΔxM_p[i]))
        ξA_p.append((mA_p[i] - 1) / (mA_p[i] + 1) - 1)
        A0_A_p_list.append((Rankine_P ** (1 + ξA_p[i])) * (1 + ξA_p[i] * sin(φ_r) + (2 / mA_p[i] - 1) * ah_av * tan(φ_r) * (2 + ξA_p[i] * (1 + sin(φ_r)))))
        if A0_A_p_list[i] < 1:
            λ_val_p = 1
        else:
            λ_val_p = 0    
        two_λ_p_list.append(2 * λ_val_p - 1)
        B1_A_p.append((two_λ_p_list[i] * 2 * c / ((1 - av) * σv_prime)) * tan(pi / 4 - φ_r / 2)
                        * (tan(pi / 4 + φ_r / 2) / tan(pi / 4 - φ_r / 2)) ** (1 + ξA_p[i]))
        e1_p_list.append((1 - A0_A_p_list[i]) / B1_A_p[i])
        e2_p_list.append((1 + A0_A_p_list[i]) / (two_λ_p_list[i] * B1_A_p[i])
                           + 2 * c / ((1 - av) * σv_prime * two_λ_p_list[i] * B1_A_p[i] * tan(φ_r)))
        α0_p_list.append(1 + (e2_p_list[i] ** 2) * (tan(φ_r) ** 2))
        b0_p_list.append(1 - (2 * two_λ_p_list[i] * e1_p_list[i] * e2_p_list[i] + e2_p_list[i] ** 2) * (tan(φ_r) ** 2))            
        twoλ_1_3α0_p_list.append(-1 / (3 * two_λ_p_list[i] * α0_p_list[i])) 
        c0_p_list.append((e1_p_list[i] ** 2 + 2 * two_λ_p_list[i] * e1_p_list[i] * e2_p_list[i]) * (tan(φ_r) ** 2))
        d0_p_list.append(-e1_p_list[i] ** 2 * (tan(φ_r) ** 2))
        D0_p_list.append(b0_p_list[i] ** 2 - 3 * α0_p_list[i] * c0_p_list[i])
        D1_p_list.append(2 * b0_p_list[i] ** 3 - 9 * α0_p_list[i] * b0_p_list[i] * c0_p_list[i] + 27 * α0_p_list[i] ** 2 * d0_p_list[i])
        IMSQRT_p_list.append((D1_p_list[i] ** 2 - 4 * (D0_p_list[i] ** 3)) ** 0.5)
        D1_sqrt_p_list.append((D1_p_list[i] - IMSQRT_p_list[i]) / 2)
        C0_p_list.append(D1_sqrt_p_list[i] ** (1 / 3))
        ζ_λ_C0_p_list.append((ζ ** λ_val_p) * C0_p_list[i])
        D0_ζλC0_p_list.append(D0_p_list[i] / ζ_λ_C0_p_list[i])
        bo_ζλC0_p_list.append(b0_p_list[i] + ζ_λ_C0_p_list[i] + D0_ζλC0_p_list[i])
        colV_colX_p_list.append(bo_ζλC0_p_list[i].real * twoλ_1_3α0_p_list[i].real)
        φm_p_list.append(asin(colV_colX_p_list[i]))
        φm_deg_p_int.append(φm_p_list[i] * (180 / pi))
        cm_p_int.append(c * tan(φm_p_list[i]) / tan(φ_r))
        Κ_ΧΕ_p_int.append((1 - two_λ_p_list[i] * sin(φm_p_list[i])) / (1 + two_λ_p_list[i] * sin(φm_p_list[i]))
                           - two_λ_p_list[i] * 2 * cm_p_int[i] * tan(pi / 4 - two_λ_p_list[i] * φm_p_list[i] / 2)
                           / (σv_prime * (1 - av)))
   # --- Build the Active Intermediate Results Table ---
    html_active = "<h3>Intermediate Active State</h3>"
    html_active += "<table border='1' style='border-collapse: collapse; width:100%;'>"
    html_active += ("<tr>"
                    "<th>Δx/Δx_max</th>"
                    "<th>φ_m [°]</th>"
                    "<th>c_m [kPa]</th>"
                    "<th>K_IA</th>"
                    "<th>Δx [mm]</th>"
                    "</tr>")
    for i in range(0, Inte_p):
        html_active += ("<tr>"
                        f"<td>{round(res_a[i], 3)}</td>"
                        f"<td>{round(φm_deg_int[i], 3)}</td>"
                        f"<td>{round(cm_int[i], 3)}</td>"
                        f"<td>{round(Κ_ΧΕ_int[i], 3)}</td>"
                        f"<td>{round(Δx_int_act[i] * 1000, 3)}</td>"
                        "</tr>")
    html_active += "</table>"

    # --- Build the Passive Intermediate Results Table ---
    html_passive = "<h3>Intermediate Passive State</h3>"
    html_passive += "<table border='1' style='border-collapse: collapse; width:100%;'>"
    html_passive += ("<tr>"
                     "<th>Δx/Δx_max</th>"
                     "<th>φ_m [°]</th>"
                     "<th>c_m [kPa]</th>"
                     "<th>K_IP</th>"
                     "<th>Δx [mm]</th>"
                     "</tr>")
    for i in range(0, Inte_p):
        html_passive += ("<tr>"
                         f"<td>{round(res_p[i], 3)}</td>"
                         f"<td>{round(φm_deg_p_int[i], 3)}</td>"
                         f"<td>{round(cm_p_int[i], 3)}</td>"
                         f"<td>{round(Κ_ΧΕ_p_int[i], 3)}</td>"
                         f"<td>{round(Δx_int_pass[i] * 1000, 3)}</td>"
                         "</tr>")
    html_passive += "</table>"

    # Combine both tables into one HTML string
    intermediate_html = html_active + "<br>" + html_passive

    # Now update the innerHTML of the element with id "intermediate-results"
    document.getElementById("intermediate-results").innerHTML = intermediate_html

def results_python(*args, **kwargs):

    σv_prime, H, c, φ_d, ah, av, Es, ν, Z, Inte_p = data()
    Active, cm_a, φm_deg_a, ΔxM_a = active(99999999999, 0)
    write_func('label_K_AE', round(Active, 3))
    write_func('label_cm_a', round(cm_a, 3))
    write_func('label_phi_m_deg_a', round(φm_deg_a, 3))
    write_func('label_DxM_a', round(ΔxM_a*1000, 3))
    write_func('label_act', round(σv_prime*Active, 3))
    At_rest, cm_o, φm_deg_o = at_rest(1, 0)
    write_func('label_At_rest', round(At_rest, 3))
    write_func('label_cm_o', round(cm_o, 3))
    write_func('label_phi_m_deg_o', round(φm_deg_o, 3))
    write_func('label_rest', round(σv_prime*At_rest, 3))
    Passive, cm_p_s, φm_deg_p_s, ΔxM_p = passive(99999999999, 0)
    write_func('label_Passive', round(Passive, 3))
    write_func('label_cm_p_s', round(cm_p_s, 3))
    write_func('label_phi_m_deg_p_s', round(φm_deg_p_s, 3))
    write_func('label_DxM_p', round(ΔxM_p*1000, 3))
    write_func('label_pas', round(σv_prime*Passive, 3))

def my_function2(*args, **kwargs): 
    σv_prime, H, c, φ_d, ah, av, Es, ν, Z, Inte_p = data()
    Δx_new = float(Element('Δx_new').element.value)
    Δx_new = Δx_new/1000
 
    Passive, cm_p_s, φm_deg_p_s, ΔxM_p = passive(99999999999, 0)
    Active, cm_a, φm_deg_a, ΔxM_a = active(99999999999, 0)
    At_rest, cm_o, φm_deg_o = at_rest(1, 0)
    Κ_ΧΕ_p_2, cm_p, φm_deg_p = int_passive(Δx_new)
    Κ_ΧΕ_int_a, cm_int_a, φm_deg_int_a = int_active(Δx_new)

    if Δx_new == 0:   
        write_func('label_text1a', "K_OE") 
        write_func('label_text2a', "c_m [kPa]") 
        write_func('label_text3a', "φ_m [°]") 
        write_func('label_text4a', "σ'_OE [kPa]")      
        write_func('label_text5a', "")
        write_func('label_value1a', round(At_rest, 3))   
        write_func('label_value2a', round(cm_o, 3)) 
        write_func('label_value3a', round(φm_deg_o, 3)) 
        write_func('label_value4a', round(σv_prime*At_rest, 3))      
        write_func('label_value5a', "")
        write_func('label_text1p', "") 
        write_func('label_text2p', "") 
        write_func('label_text3p', "")    
        write_func('label_text4p', "") 
        write_func('label_text5p', "") 
        write_func('label_value1p', "")   
        write_func('label_value2p', "")  
        write_func('label_value3p', "")
        write_func('label_value4p', "")     
        write_func('label_value5p', "")   
    if Δx_new > ΔxM_p:
        write_func('label_text1a', "K_AE") 
        write_func('label_text2a', "c_m [kPa]") 
        write_func('label_text3a', "φ_m [°]") 
        write_func('label_text4a', "Δx_max Active")
        write_func('label_text5a', "σ'_AE [kPa]")    
        write_func('label_value1a', round(Active, 3))    
        write_func('label_value2a', round(cm_a, 3))
        write_func('label_value3a', round(φm_deg_a, 3))
        write_func('label_value4a', round(ΔxM_a*1000, 3))
        write_func('label_value5a', round(σv_prime*Active, 3)) 
        write_func('label_text1p', "Κ_ΡΕ") 
        write_func('label_text2p', "c_m [kPa]") 
        write_func('label_text3p', "φ_m [°]")    
        write_func('label_text4p', "Δx_max Passive") 
        write_func('label_text5p', "σ'_PE [kPa]") 
        write_func('label_value1p', round(Passive, 3))  
        write_func('label_value2p', round(cm_p_s, 3))   
        write_func('label_value3p', round(φm_deg_p_s, 3)) 
        write_func('label_value4p', round(ΔxM_p*1000, 3))    
        write_func('label_value5p', round(σv_prime*Passive, 3)) 
    if 0 < Δx_new <= ΔxM_a:  
        write_func('label_text1a', "K_IA") 
        write_func('label_text2a', "c_m [kPa]") 
        write_func('label_text3a', "φ_m [°]") 
        write_func('label_text4a', "Δx_max Active")
        write_func('label_text5a', "σ'_IA [kPa]") 
        write_func('label_value1a', round(Κ_ΧΕ_int_a, 3))  
        write_func('label_value2a', round(cm_int_a, 3)) 
        write_func('label_value3a', round(φm_deg_int_a, 3)) 
        write_func('label_value4a', round(ΔxM_a*1000, 3)) 
        write_func('label_value5a', round(σv_prime*Κ_ΧΕ_int_a, 3))
        write_func('label_text1p', "K_IP") 
        write_func('label_text2p', "c_m [kPa]") 
        write_func('label_text3p', "φ_m [°]")    
        write_func('label_text4p', "Δx_max Passive")
        write_func('label_text5p', "σ'_IP [kPa]") 
        write_func('label_value1p', round(Κ_ΧΕ_p_2, 3))   
        write_func('label_value2p', round(cm_p, 3))  
        write_func('label_value3p', round(φm_deg_p, 3))   
        write_func('label_value4p', round(ΔxM_p*1000, 3))   
        write_func('label_value5p', round(σv_prime*Κ_ΧΕ_p_2, 3)) 
    if ΔxM_a < Δx_new <= ΔxM_p:
        write_func('label_text1a', "K_AE") 
        write_func('label_text2a', "c_m [kPa]") 
        write_func('label_text3a', "φ_m [°]") 
        write_func('label_text4a', "Δx_max Active")
        write_func('label_text5a', "σ'_AE [kPa]")
        write_func('label_value1a', round(Active, 3)) 
        write_func('label_value2a', round(cm_a, 3)) 
        write_func('label_value3a', round(φm_deg_a, 3)) 
        write_func('label_value4a', round(ΔxM_a*1000, 3)) 
        write_func('label_value5a', round(σv_prime*Active, 3))
        write_func('label_text1p', "K_IP") 
        write_func('label_text2p', "c_m [kPa]") 
        write_func('label_text3p', "φ_m [°]")    
        write_func('label_text4p', "Δx_max Passive") 
        write_func('label_text5p', "σ'_IP [kPa]") 
        write_func('label_value1p', round(Κ_ΧΕ_p_2, 3)) 
        write_func('label_value2p', round(cm_p, 3))
        write_func('label_value3p', round(φm_deg_p, 3)) 
        write_func('label_value4p', round(ΔxM_p*1000, 3))
        write_func('label_value5p', round(σv_prime*Κ_ΧΕ_p_2, 3))
