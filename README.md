 © 2025 **Marcela Veiga** — Todos os direitos reservados.  
> Este repositório contém código e documentação do projeto **Bio Sensor App**,  
> desenvolvido para análise molecular de amostras biológicas via espectroscopia Raman.  
> O sistema, sua arquitetura e pipeline de processamento são propriedade intelectual da autora.  
> O uso, cópia ou redistribuição deste material é proibido sem autorização expressa.  
> O uso acadêmico é permitido mediante citação adequada:
>
> **Veiga, M. (2025). Bio Sensor App: Plataforma Integrada para Análise Molecular via Espectroscopia Raman e Supabase.**

# Bio Sensor App — Plataforma Raman Integrada com Supabase

Plataforma web em **Streamlit** conectada ao **Supabase** para cadastro de pacientes, análise de espectros Raman e otimização por IA.  
Projetada para aplicações em **biossensoriamento**, **análise molecular de sangue** e **caracterização de superfícies**.  

Desenvolvido em Python, totalmente modular e compatível com **deploy via GitHub + Streamlit Cloud**.

---

## Funcionalidades Principais
1 Cadastro de Pacientes e Amostras
- Importação automática de pacientes via **Google Forms (CSV)**.  
- Criação manual de pacientes diretamente na interface.  
- Importação em lote de **espectros Raman brutos** a partir de arquivos `.zip`, com:
  - Mapeamento automático (nome do arquivo → paciente).  
  - Suporte a `mapping.csv` (colunas: `filename`, `full_name`, `email`, `cpf`).  
  - Criação automática de pacientes e amostras no Supabase.  
  - Processamento de **20 arquivos por vez** para evitar erros e timeouts.

2 Espectrometria Raman (Análise Molecular)
- Processamento completo via módulo `raman_processing_v2.py`:  
  - Remoção de substrato e baseline.  
  - Suavização com **Savitzky-Golay filter**.  
  - Detecção e ajuste **Lorentziano** de picos.  
  - Normalização e construção de espectro corrigido.  
  - Correlação dos picos com **grupos moleculares** (hemoglobina, proteínas, etc.).  
- Visualização:
  - Gráfico principal (ajuste total + baseline + picos).  
  - Gráfico de **resíduo** (diferença entre modelo e dados experimentais).  
  - Download dos espectros e picos detectados em CSV.  

3 Otimização (IA)
- Treinamento de modelos **Random Forest** para classificar espectros rotulados.  
- Upload de CSV de treino com coluna `label`.  
- Cálculo automático de acurácia e importância das features.  
- Upload de novos dados para previsão automática.


## Estrutura do Projeto

bio_sensor_app/
│
├── app.py                      # Aplicativo principal (3 abas)
├── raman_processing_v2.py      # Pipeline de análise Raman (baseline, picos, plot)
├── requirements.txt            # Dependências do Python
├── README.md                   # Este arquivo
└── utils/
    └── batch_import_zip.py     # Importador em lotes (ZIP de espectros)

## Créditos

Desenvolvido por **Marcela Veiga**  
Baseado em protocolos de caracterização molecular de sangue via Raman sobre substratos de papel.  
Integração Supabase + Streamlit

## Licença

Este projeto está protegido por direitos autorais.  
O uso, cópia ou redistribuição são proibidos sem autorização expressa da autora.  
O uso acadêmico é permitido mediante citação:

**Veiga, M. (2025). Bio Sensor App: Plataforma Integrada para Análise Molecular via Espectroscopia Raman e Supabase.**
