import taichi as ti
import numpy as np

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import *
from src.physics_model.consititutive_model.infinitesimal_strain.ElasPlasticity import PlasticMaterial
from src.utils.constants import FTOL
from src.utils.ObjectIO import DictIO
from src.utils.VectorFunction import voigt_tensor_trace, voigt_tensor_dot
import src.utils.GlobalVariable as GlobalVariable


@ti.data_oriented
class NorSandModel(PlasticMaterial):
    """
    NorSand constitutive model.

    Referências:
      - Jefferies (1993) "Nor-Sand: a simple critical state model for sand"
      - Jefferies & Been (2006) "Soil Liquefaction: A Critical State Approach"
      - Borja & Andrade (CMAME, 2006)
      - Implementação de referência: Choo Group / GeoWarp (Yidong Zhao, 2024)
        https://github.com/choo-group/GeoWarp/blob/main/problems/material_tests/triaxial_NorSand.py

    Convenção GeoTaichi: tração positiva.
        p_GT = SphericalTensor(σ) ≤ 0   (compressão → p_GT negativo)
        p_c  = -p_GT > 0                (compressão positiva, uso interno)
        pi   > 0                        (pressão de imagem, compressão positiva)

    Variável interna plástica única: pi (pressão de imagem).

    Convenção da CSL adotada (normalizada):
        v_c(p) = vc0 - λ · ln(p / p_ref)
        ψ_i    = v - v_c(pi) = v - vc0 + λ · ln(pi / p_ref)
    onde vc0 é o volume específico crítico em p = p_ref.

    NOTA SOBRE UNIDADES: o GeoTaichi opera em SI (Pa). Portanto:
      - G0:    Pa     (módulo de cisalhamento na referência)
      - p_ref: Pa     (tipicamente 1.0e5 Pa = 100 kPa)
      - h:     adimensional
      - vc0:   adimensional (volume específico)
      - lambda, kappa: adimensional (coeficientes da CSL/swelling)

    material_params (vetor Taichi, montado no início de cada passo):
        [0] = void_ratio  (FIXO durante o substepping)
        [1] = pi          (do início do passo)
        [2] = p_c         (= -p_GT do início do passo)

    internal_vars (vetor Taichi, evolui durante o substepping):
        [0] = pi
    """

    def __init__(self, material_type="Solid", configuration="UL",
                 solver_type="Explicit", stress_integration="SubStepping"):
        super().__init__(material_type, configuration, solver_type, stress_integration)
        # Defaults (sobrescritos por model_initialize)
        self.G0       = 1.0e7
        self.kappa    = 0.005
        self.lmbda    = 0.05
        self.M        = 1.30
        self.N        = 0.3
        self.beta_dil = 1.0
        self.vc0      = 1.85
        self.v0       = 1.80
        self.h        = 100.
        self.p_ref    = 1.0e5     # 100 kPa em Pa
        self.density  = 1900.
        self.is_soft  = True
        self.max_sound_speed = 0.

    # =========================================================================
    # Inicialização Python
    # =========================================================================
    def model_initialize(self, material):
        self.density  = DictIO.GetAlternative(material, 'Density',  1900.)
        self.G0       = DictIO.GetEssential(material,   'G0')
        self.kappa    = DictIO.GetEssential(material,   'kappa')
        self.lmbda    = DictIO.GetEssential(material,   'lambda')
        self.M        = DictIO.GetEssential(material,   'M')
        self.N        = DictIO.GetEssential(material,   'N')
        self.beta_dil = DictIO.GetEssential(material,   'beta')
        self.vc0      = DictIO.GetEssential(material,   'vc0')
        self.v0       = DictIO.GetEssential(material,   'v0')
        self.h        = DictIO.GetEssential(material,   'h')
        self.p_ref    = DictIO.GetAlternative(material, 'p_ref', 1.0e5)
        self.add_coupling_material(material)

    def get_sound_speed(self):
        return 0.

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print("Constitutive model = NorSand (Jefferies 1993; Borja & Andrade 2006)")
        print(f"Model ID:           {materialID}")
        print(f"Density:            {self.density} kg/m³")
        print(f"G0:                 {self.G0:.3e} Pa")
        print(f"kappa (κ):          {self.kappa}")
        print(f"lambda (λ):         {self.lmbda}")
        print(f"M:                  {self.M}")
        print(f"N:                  {self.N}")
        print(f"beta (regra fluxo): {self.beta_dil}")
        print(f"vc0 (em p_ref):     {self.vc0}")
        print(f"v0 (= 1+e0):        {self.v0}  →  e0 = {self.v0-1:.4f}")
        print(f"h (hardening):      {self.h}")
        print(f"p_ref:              {self.p_ref:.3e} Pa")
        # ψ inicial estimado em p = p_ref (referência)
        psi_ref = self.v0 - self.vc0
        print(f"ψ inicial (em p_ref): {psi_ref:+.4f}  ", end="")
        if psi_ref > 0.05:
            print("[fofo — propenso a contração/liquefação estática]")
        elif psi_ref < -0.05:
            print("[denso — propenso a dilatância]")
        else:
            print("[próximo da CSL]")
        print()

    # =========================================================================
    # Variáveis de estado por partícula
    # =========================================================================
    def define_state_vars(self):
        return {
            'pi':         float,
            'void_ratio': float,
        }

    def get_lateral_coefficient(self, start_index, end_index, materialID, stateVars):
        sin_phi = 3. * self.M / (6. + self.M)
        return np.repeat(1. - sin_phi, end_index - start_index)

    # =========================================================================
    # Inicialização pós-campo gravitacional
    # NC-aproximada: pi0 = p_c0;  void_ratio = v0 - 1
    #
    # OBSERVAÇÃO: para um setup mais consistente seria preciso resolver
    # f(p, q, pi) = 0 partícula a partícula. Por ora, usamos pi = p_c
    # (estado isotrópico equivalente), aceitando um leve transient inicial.
    # =========================================================================
    @ti.func
    def _initialize_vars_update_lagrangian(self, np, particle, stateVars):
        p_GT = SphericalTensor(particle[np].stress)
        p_c  = ti.max(-p_GT, 1e-2)
        stateVars[np].pi         = p_c
        stateVars[np].void_ratio = self.v0 - 1.0

    # =========================================================================
    # Helpers internos
    # =========================================================================
    @ti.func
    def _eta_image(self, p_c, pi):
        """Razão de tensões de imagem η_i(p_c, pi)."""
        p_eff  = ti.max(p_c, 1e-4)
        pi_eff = ti.max(pi,  1e-4)
        result = 0.0
        if ti.abs(self.N) < 1e-10:
            result = self.M * (1.0 + ti.log(pi_eff / p_eff))
        else:
            result = (self.M / self.N) * (
                1.0 - (1.0 - self.N) * (p_eff / pi_eff) ** (self.N / (1.0 - self.N))
            )
        return result

    @ti.func
    def _pi_star(self, p_c, pi, void_ratio):
        """
        Pressão de imagem alvo na linha de estado crítico — Borja & Andrade (2006).

        ψ_i = v - vc0 + λ · ln(pi / p_ref)
            (avaliado em pi, NÃO em p_c — diferença essencial em relação ao
             que estava no código anterior)

        ᾱ = -3.5 / β  (parâmetro de dilatância de Jefferies)

        Para N → 0:   pi* = p_c · exp(ᾱ · ψ_i / M)
        Para N ≠ 0:   pi* = p_c · (1 - ᾱ · ψ_i · N / M)^((N-1)/N)
        """
        pi_eff    = ti.max(pi, 1e-4)
        v         = 1.0 + void_ratio
        # ψ_i avaliado em pi, normalizado por p_ref
        psi_i     = v - self.vc0 + self.lmbda * ti.log(pi_eff / self.p_ref)
        alpha_bar = -3.5 / self.beta_dil

        result = 0.0
        if ti.abs(self.N) < 1e-10:
            # Caso limite N → 0
            result = p_c * ti.exp(alpha_bar * psi_i / self.M)
        else:
            # Caso geral N ≠ 0 (power law)
            arg      = 1.0 - alpha_bar * psi_i * self.N / self.M
            arg_safe = ti.max(arg, 1e-6)
            result   = p_c * ti.pow(arg_safe, (self.N - 1.0) / self.N)

        return ti.max(result, 1e-4)

    # =========================================================================
    # Módulos elásticos
    #   K = v · p_c / κ
    #   G = G0 · √(p_c / p_ref)
    # =========================================================================
    @ti.func
    def ComputeElasticModulus(self, stress, material_params):
        void_ratio = material_params[0]
        p_GT       = SphericalTensor(stress)
        p_c        = ti.max(-p_GT, 1e-4)
        v          = 1.0 + void_ratio
        K          = ti.max(v * p_c / self.kappa, 100.0)
        G          = self.G0 * ti.sqrt(p_c / self.p_ref)
        return K, G

    @ti.func
    def ComputeElasticStress(self, alpha, dstrain, stress, material_params):
        K, G = self.ComputeElasticModulus(stress, material_params)
        stress += ElasticTensorMultiplyVector(alpha * dstrain, K, G)
        return stress

    # =========================================================================
    # Invariantes
    # =========================================================================
    @ti.func
    def ComputeStressInvariants(self, stress):
        p = SphericalTensor(stress)
        q = EquivalentDeviatoricStress(stress)
        return p, q

    # =========================================================================
    # Função de plastificação:  f = q - η_i(p_c, pi) · p_c
    # =========================================================================
    @ti.func
    def ComputeYieldFunction(self, stress, internal_vars, material_params):
        pi   = internal_vars[0]
        p_GT = SphericalTensor(stress)
        p_c  = ti.max(-p_GT, 1e-4)
        q    = EquivalentDeviatoricStress(stress)
        eta  = self._eta_image(p_c, pi)
        return q - eta * p_c

    @ti.func
    def ComputeYieldState(self, stress, internal_vars, material_params):
        f = self.ComputeYieldFunction(stress, internal_vars, material_params)
        return f > -FTOL, f

    # =========================================================================
    # ∂f/∂σ
    #
    # Identidade algébrica (ver Borja & Andrade 2006, eq. 24):
    #   ∂f/∂p_GT = (η - M)/(1-N)  =  η - M·(p_c/pi)^(N/(1-N))
    # Mantemos a segunda forma por equivalência e estabilidade numérica.
    # =========================================================================
    @ti.func
    def ComputeDfDsigma(self, yield_state, stress, internal_vars, material_params):
        pi    = internal_vars[0]
        p_GT  = SphericalTensor(stress)
        p_c   = ti.max(-p_GT, 1e-4)
        pi_e  = ti.max(pi, 1e-4)
        eta   = self._eta_image(p_c, pi_e)
        exp_N = self.N / (1.0 - self.N + 1e-12)
        dfdp_GT = eta - self.M * ti.pow(p_c / pi_e, exp_N)
        return DqDsigma(stress) + dfdp_GT * DpDsigma()

    # =========================================================================
    # ∂g/∂σ — regra de fluxo (Rowe stress-dilatancy)
    #
    # D = M - η_curr  (η_curr = q/p_c instantâneo)
    # ∂g/∂p_GT = η_curr - M
    # =========================================================================
    @ti.func
    def ComputeDgDsigma(self, yield_state, stress, internal_vars, material_params):
        p_GT     = SphericalTensor(stress)
        p_c      = ti.max(-p_GT, 1e-4)
        q        = EquivalentDeviatoricStress(stress)
        eta_curr = q / ti.max(p_c, 1e-4)
        dgdp_GT  = eta_curr - self.M
        return DqDsigma(stress) + dgdp_GT * DpDsigma()

    # =========================================================================
    # Módulo de hardening H (negativo para hardening — convenção do framework)
    # H = (∂f/∂pi) · (dpi/dλ)  com  dpi/dλ = h · (pi* - pi)
    # =========================================================================
    @ti.func
    def ComputePlasticModulus(self, yield_state, dgdsigma, stress,
                               internal_vars, state_vars, material_params):
        pi         = internal_vars[0]
        # Usa o void_ratio fixo do início do passo (estabilidade no substepping)
        void_ratio = state_vars.void_ratio
        p_c        = ti.max(material_params[2], 1e-4)
        pi_e       = ti.max(pi, 1e-4)

        # CORREÇÃO: _pi_star agora recebe pi (não só p_c) para avaliar ψ_i
        pi_star = self._pi_star(p_c, pi_e, void_ratio)
        exp_N   = self.N / (1.0 - self.N + 1e-12)
        dfdpi   = -self.M * p_c * ti.pow(p_c / pi_e, exp_N) / pi_e

        H = dfdpi * self.h * (pi_star - pi)

        # Limite físico para impedir snap-back numérico
        K, G = self.ComputeElasticModulus(stress, material_params)
        H_min = -0.9 * (3.0 * G + K)
        return ti.max(H, H_min)

    # =========================================================================
    # Incremento da variável interna plástica
    #   dpi = h · (pi* - pi) · dλ
    # =========================================================================
    @ti.func
    def ComputeInternalVariables(self, dlambda, dgdsigma, internal_vars, material_params):
        pi         = internal_vars[0]
        void_ratio = material_params[0]
        p_c        = ti.max(material_params[2], 1e-4)

        # CORREÇÃO: _pi_star agora recebe pi
        pi_star = self._pi_star(p_c, pi, void_ratio)
        dpi     = self.h * (pi_star - pi) * dlambda
        return ti.Vector([dpi])

    # =========================================================================
    # Empacotamento / desempacotamento de estado
    # =========================================================================
    @ti.func
    def GetMaterialParameter(self, stress, state_vars):
        void_ratio = state_vars.void_ratio
        pi         = state_vars.pi
        p_GT       = SphericalTensor(stress)
        p_c        = ti.max(-p_GT, 1e-4)
        return ti.Vector([void_ratio, pi, p_c])

    @ti.func
    def GetInternalVariables(self, state_vars):
        return ti.Vector([state_vars.pi])

    @ti.func
    def UpdateInternalVariables(self, np, internal_vars, stateVars):
        stateVars[np].pi = ti.max(internal_vars[0], 1e-4)

    # =========================================================================
    # Atualização do void_ratio ao fim de cada passo (fora do substepping).
    # Aproximação via linha NC + swelling line:
    #   e_NC(pi) = (vc0 - 1) - λ · ln(pi / p_ref)
    #   e_atual  = e_NC + κ · ln(pi / p_c)
    # =========================================================================
    @ti.func
    def UpdateStateVariables(self, np, stress, internal_vars, stateVars):
        p_GT    = SphericalTensor(stress)
        p_c_new = ti.max(-p_GT, 1e-4)
        pi_new  = ti.max(stateVars[np].pi, 1e-4)

        # CORREÇÃO: log normalizado por p_ref (consistente com _pi_star)
        e_nc  = (self.vc0 - 1.0) - self.lmbda * ti.log(pi_new / self.p_ref)
        e_new = e_nc + self.kappa * ti.log(pi_new / p_c_new)
        e_new = ti.max(e_new, 0.05)  # piso físico
        stateVars[np].void_ratio = e_new

    @ti.func
    def get_current_material_parameter(self, state_vars):
        return state_vars.pi, state_vars.void_ratio

    @ti.func
    def compute_elastic_tensor(self, np, current_stress, stateVars):
        material_params = self.GetMaterialParameter(current_stress, stateVars[np])
        K, G = self.ComputeElasticModulus(current_stress, material_params)
        return ComputeElasticStiffnessTensor(K, G)
