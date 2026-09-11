import io
import json

from moid.adapters import ollama_client


def test_structured_reply_in_wrong_field_is_recovered_only_when_enabled(monkeypatch):
    requests = []
    def response(request, **kwargs):
        requests.append(json.loads(request.data))
        return io.BytesIO(json.dumps({'message': {'content': '', 'thinking':
                            '{"match":"yes","reason":"Visible details agree"}'}}).encode())
    monkeypatch.setattr(ollama_client, 'urlopen', response)
    args = dict(host='http://localhost', model='test', messages=[], response_format='json',
                think=False, num_gpu=0, num_thread=4)
    assert ollama_client.ollama_chat(**args) == ''
    recovered = ollama_client.ollama_chat(**args, structured_response_fallback=True)
    assert json.loads(recovered)['match'] == 'yes'
    assert requests[-1]['think'] is False
    assert requests[-1]['options']['num_gpu'] == 0
    assert requests[-1]['options']['num_thread'] == 4


def test_free_form_thinking_is_not_a_response(monkeypatch):
    monkeypatch.setattr(ollama_client, 'urlopen', lambda *args, **kw:
                        io.BytesIO(b'{"message":{"content":"","thinking":"Maybe the same car"}}'))
    assert ollama_client.ollama_chat(host='http://localhost', model='test', messages=[],
        response_format='json', think=False, structured_response_fallback=True) == ''
