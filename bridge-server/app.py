"""
Bridge Server - Alexa + Home Assistant + Claude AI
Servidor intermediário que processa comandos da Alexa usando Claude AI
e executa ações no Home Assistant.
"""

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import anthropic
import httpx
import os
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
import json

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicialização
app = FastAPI(title="Alexa HA AI Bridge")

# Configurações (use variáveis de ambiente)
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
HA_URL = os.getenv("HOME_ASSISTANT_URL", "http://localhost:8123")
HA_TOKEN = os.getenv("HOME_ASSISTANT_TOKEN", "")
BRIDGE_API_KEY = os.getenv("BRIDGE_API_KEY", "your-secure-key-here")

# Cliente Anthropic
claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

# Cliente HTTP para Home Assistant
http_client = httpx.AsyncClient()


class AlexaRequest(BaseModel):
    """Modelo de request vindo da Alexa Lambda"""
    command: str
    context: Optional[Dict[str, Any]] = {}
    user_id: Optional[str] = None


class AlexaResponse(BaseModel):
    """Modelo de response para Alexa"""
    speech: str
    should_end_session: bool = False
    card_title: Optional[str] = None
    card_content: Optional[str] = None


# ============================================================================
# TOOLS PARA CLAUDE - Definições das funções que Claude pode chamar
# ============================================================================

TOOLS = [
    {
        "name": "get_home_state",
        "description": "Obtém o estado atual de dispositivos/sensores no Home Assistant. Use para verificar se luzes estão acesas, temperatura atual, estado de portas/janelas, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lista de entity IDs específicos (ex: ['light.sala', 'sensor.temperatura']). Deixe vazio para obter todos."
                },
                "domain": {
                    "type": "string",
                    "description": "Filtrar por domínio específico: 'light', 'switch', 'sensor', 'climate', 'lock', 'cover', etc."
                }
            }
        }
    },
    {
        "name": "control_device",
        "description": "Liga, desliga ou ajusta um dispositivo específico no Home Assistant.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "Entity ID do dispositivo (ex: 'light.sala', 'switch.ventilador')"
                },
                "action": {
                    "type": "string",
                    "enum": ["turn_on", "turn_off", "toggle"],
                    "description": "Ação a executar"
                },
                "attributes": {
                    "type": "object",
                    "description": "Atributos adicionais (brightness: 0-255, rgb_color: [r,g,b], temperature: valor)"
                }
            },
            "required": ["entity_id", "action"]
        }
    },
    {
        "name": "control_climate",
        "description": "Controla ar condicionado, aquecedor ou termostato.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "Entity ID do dispositivo de clima"
                },
                "temperature": {
                    "type": "number",
                    "description": "Temperatura desejada em Celsius"
                },
                "hvac_mode": {
                    "type": "string",
                    "enum": ["heat", "cool", "heat_cool", "auto", "off", "fan_only", "dry"],
                    "description": "Modo de operação"
                },
                "fan_mode": {
                    "type": "string",
                    "description": "Modo do ventilador (low, medium, high, auto)"
                }
            },
            "required": ["entity_id"]
        }
    },
    {
        "name": "execute_scene",
        "description": "Executa uma cena predefinida no Home Assistant (ex: 'scene.cinema', 'scene.jantar').",
        "input_schema": {
            "type": "object",
            "properties": {
                "scene_id": {
                    "type": "string",
                    "description": "Entity ID da cena"
                }
            },
            "required": ["scene_id"]
        }
    },
    {
        "name": "call_service",
        "description": "Chama qualquer serviço do Home Assistant de forma genérica.",
        "input_schema": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Domínio do serviço (ex: 'light', 'switch', 'notify')"
                },
                "service": {
                    "type": "string",
                    "description": "Nome do serviço (ex: 'turn_on', 'toggle')"
                },
                "entity_id": {
                    "type": "string",
                    "description": "Entity ID alvo (opcional)"
                },
                "data": {
                    "type": "object",
                    "description": "Dados adicionais para o serviço"
                }
            },
            "required": ["domain", "service"]
        }
    },
    {
        "name": "get_history",
        "description": "Obtém histórico de estados de entidades para análise de padrões.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lista de entity IDs"
                },
                "hours": {
                    "type": "integer",
                    "description": "Quantas horas de histórico buscar (padrão: 24)",
                    "default": 24
                }
            },
            "required": ["entity_ids"]
        }
    }
]


# ============================================================================
# IMPLEMENTAÇÃO DOS TOOLS - Comunicação com Home Assistant
# ============================================================================

async def execute_tool(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """Executa um tool e retorna o resultado"""
    
    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        if tool_name == "get_home_state":
            # GET /api/states ou /api/states/<entity_id>
            if tool_input.get("entity_ids"):
                states = []
                for entity_id in tool_input["entity_ids"]:
                    url = f"{HA_URL}/api/states/{entity_id}"
                    response = await http_client.get(url, headers=headers)
                    response.raise_for_status()
                    states.append(response.json())
                return {"states": states}
            else:
                url = f"{HA_URL}/api/states"
                response = await http_client.get(url, headers=headers)
                response.raise_for_status()
                all_states = response.json()
                
                # Filtrar por domínio se especificado
                if tool_input.get("domain"):
                    domain = tool_input["domain"]
                    all_states = [s for s in all_states if s["entity_id"].startswith(f"{domain}.")]
                
                return {"states": all_states}
        
        elif tool_name == "control_device":
            entity_id = tool_input["entity_id"]
            action = tool_input["action"]
            domain = entity_id.split(".")[0]
            
            url = f"{HA_URL}/api/services/{domain}/{action}"
            data = {"entity_id": entity_id}
            
            # Adicionar atributos se fornecidos
            if tool_input.get("attributes"):
                data.update(tool_input["attributes"])
            
            response = await http_client.post(url, headers=headers, json=data)
            response.raise_for_status()
            return {"success": True, "action": action, "entity": entity_id}
        
        elif tool_name == "control_climate":
            entity_id = tool_input["entity_id"]
            url = f"{HA_URL}/api/services/climate/set_temperature"
            
            data = {"entity_id": entity_id}
            
            if "temperature" in tool_input:
                data["temperature"] = tool_input["temperature"]
            
            if "hvac_mode" in tool_input:
                # Primeiro define o modo
                mode_url = f"{HA_URL}/api/services/climate/set_hvac_mode"
                mode_data = {
                    "entity_id": entity_id,
                    "hvac_mode": tool_input["hvac_mode"]
                }
                await http_client.post(mode_url, headers=headers, json=mode_data)
            
            if "temperature" in tool_input:
                response = await http_client.post(url, headers=headers, json=data)
                response.raise_for_status()
            
            return {"success": True, "entity": entity_id}
        
        elif tool_name == "execute_scene":
            scene_id = tool_input["scene_id"]
            url = f"{HA_URL}/api/services/scene/turn_on"
            data = {"entity_id": scene_id}
            
            response = await http_client.post(url, headers=headers, json=data)
            response.raise_for_status()
            return {"success": True, "scene": scene_id}
        
        elif tool_name == "call_service":
            domain = tool_input["domain"]
            service = tool_input["service"]
            url = f"{HA_URL}/api/services/{domain}/{service}"
            
            data = {}
            if tool_input.get("entity_id"):
                data["entity_id"] = tool_input["entity_id"]
            if tool_input.get("data"):
                data.update(tool_input["data"])
            
            response = await http_client.post(url, headers=headers, json=data)
            response.raise_for_status()
            return {"success": True}
        
        elif tool_name == "get_history":
            entity_ids = tool_input["entity_ids"]
            hours = tool_input.get("hours", 24)
            
            # Calcular timestamp
            from datetime import timedelta
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            histories = []
            for entity_id in entity_ids:
                url = f"{HA_URL}/api/history/period/{start_time.isoformat()}"
                params = {"filter_entity_id": entity_id}
                
                response = await http_client.get(url, headers=headers, params=params)
                response.raise_for_status()
                histories.append({
                    "entity_id": entity_id,
                    "history": response.json()
                })
            
            return {"histories": histories}
        
        else:
            return {"error": f"Tool desconhecido: {tool_name}"}
            
    except Exception as e:
        logger.error(f"Erro ao executar tool {tool_name}: {str(e)}")
        return {"error": str(e)}


# ============================================================================
# SYSTEM PROMPT PARA CLAUDE
# ============================================================================

def build_system_prompt() -> str:
    """Constrói o system prompt com contexto da casa"""
    
    now = datetime.now()
    
    return f"""Você é o assistente de casa inteligente. O usuário está controlando sua casa via Alexa.

CONTEXTO ATUAL:
- Data e hora: {now.strftime('%d/%m/%Y %H:%M')} ({now.strftime('%A').lower()})
- Localização: Contagem, Minas Gerais, Brasil
- Casa equipada com Home Assistant

SUAS CAPACIDADES:
Você tem acesso a tools que permitem:
1. get_home_state: Consultar estado de dispositivos e sensores
2. control_device: Ligar/desligar/ajustar dispositivos
3. control_climate: Controlar temperatura e climatização
4. execute_scene: Ativar cenas predefinidas
5. call_service: Executar qualquer serviço do Home Assistant
6. get_history: Analisar histórico de uso

DIRETRIZES IMPORTANTES:
1. SEMPRE use os tools para obter informações reais - NUNCA invente estados
2. Seja CONCISO - suas respostas serão faladas pela Alexa (máximo 2-3 frases)
3. Use linguagem NATURAL e AMIGÁVEL
4. Para comandos AMBÍGUOS, escolha a interpretação mais provável baseada no contexto
5. Para ações IMPORTANTES (trancar, alarmes), confirme com o usuário
6. Se NÃO TIVER CERTEZA, pergunte claramente

EXEMPLOS DE INTERPRETAÇÃO CONTEXTUAL:
- "está escuro" → get_home_state para verificar luzes, depois acender as necessárias
- "tá frio" → get_home_state da temperatura, depois ajustar clima
- "prepare para o jantar" → Sequência: ajustar luzes da sala de jantar (warm, 70%), definir temperatura confortável (22°C), possivelmente executar scene.jantar se existir
- "esqueci algo aberto?" → get_home_state de todos os sensores de porta/janela
- "boa noite" → Desligar luzes (exceto quarto), trancar portas, ajustar temperatura para sleep (18°C), ativar modo noturno se disponível
- "modo cinema" → Escurecer luzes da sala, fechar cortinas se houver, executar scene.cinema

FORMATO DE RESPOSTA:
- Seja direto e objetivo
- Use primeira pessoa ("Eu liguei as luzes" ou "Ajustei a temperatura")
- Confirme ações realizadas
- Se algo falhou, explique brevemente

EXEMPLO DE INTERAÇÃO:
User: "Alexa, prepare a casa para dormir"
You: [usa get_home_state para ver estado atual]
     [usa control_device para desligar luzes]
     [usa control_climate para ajustar temperatura]
     [usa control_device para trancar portas se houver]
     Response: "Prontinho! Apaguei as luzes, ajustei a temperatura para 18 graus e tranquei as portas. Boa noite!"
"""


# ============================================================================
# PROCESSAMENTO DE COMANDOS COM CLAUDE
# ============================================================================

async def process_command_with_claude(command: str, context: Dict) -> str:
    """Processa comando usando Claude AI com tools"""
    
    logger.info(f"Processando comando: {command}")
    
    # Construir mensagem do usuário
    user_message = f"O usuário disse via Alexa: '{command}'"
    
    # Adicionar contexto se disponível
    if context:
        user_message += f"\n\nContexto adicional: {json.dumps(context, ensure_ascii=False)}"
    
    try:
        # Chamar Claude API com tool use
        response = claude_client.messages.create(
            model="claude-sonnet-4-20250514",  # Ou use haiku para respostas mais rápidas
            max_tokens=4096,
            system=build_system_prompt(),
            tools=TOOLS,
            messages=[{
                "role": "user",
                "content": user_message
            }]
        )
        
        # Processar response e tool calls
        final_response = ""
        
        while response.stop_reason == "tool_use":
            # Executar todos os tool calls
            tool_results = []
            
            for content_block in response.content:
                if content_block.type == "tool_use":
                    tool_name = content_block.name
                    tool_input = content_block.input
                    
                    logger.info(f"Executando tool: {tool_name} com input: {tool_input}")
                    
                    # Executar o tool
                    result = await execute_tool(tool_name, tool_input)
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": json.dumps(result, ensure_ascii=False)
                    })
            
            # Continuar a conversa com os resultados dos tools
            response = claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=build_system_prompt(),
                tools=TOOLS,
                messages=[
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": response.content},
                    {"role": "user", "content": tool_results}
                ]
            )
        
        # Extrair resposta final
        for content_block in response.content:
            if hasattr(content_block, "text"):
                final_response += content_block.text
        
        logger.info(f"Resposta final: {final_response}")
        return final_response.strip()
        
    except Exception as e:
        logger.error(f"Erro ao processar com Claude: {str(e)}")
        return f"Desculpe, tive um problema ao processar seu pedido: {str(e)}"


# ============================================================================
# ENDPOINTS DA API
# ============================================================================

@app.post("/process", response_model=AlexaResponse)
async def process_alexa_command(
    request: AlexaRequest,
    x_api_key: str = Header(None)
):
    """
    Endpoint principal que recebe comandos da Alexa Lambda
    """
    
    # Validar API key
    if x_api_key != BRIDGE_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Processar comando com Claude
    response_text = await process_command_with_claude(
        command=request.command,
        context=request.context or {}
    )
    
    return AlexaResponse(
        speech=response_text,
        should_end_session=True,
        card_title="Casa Inteligente",
        card_content=response_text
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "home_assistant": HA_URL,
        "claude_configured": bool(CLAUDE_API_KEY)
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Alexa + Home Assistant + AI Bridge",
        "version": "1.0.0",
        "endpoints": {
            "POST /process": "Processar comando da Alexa",
            "GET /health": "Health check",
        }
    }


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Executado ao iniciar o servidor"""
    logger.info("🚀 Bridge Server iniciado")
    logger.info(f"Home Assistant: {HA_URL}")
    logger.info(f"Claude API: {'✓ Configurado' if CLAUDE_API_KEY else '✗ Não configurado'}")


@app.on_event("shutdown")
async def shutdown_event():
    """Executado ao desligar o servidor"""
    await http_client.aclose()
    logger.info("👋 Bridge Server desligado")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
