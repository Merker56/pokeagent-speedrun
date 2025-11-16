from agent.vlm import VertexBackend

def main():
    print("Initializing Vertex backend…")
    vlm = VertexBackend(model_name="gemini-2.5-flash")

    print("Sending test prompt to Gemini…")
    try:
        response = vlm.get_text_query("List the first three Pokémon in the Pokédex.", module_name="test")
        print("Response from Vertex:")
        print(response)
    except Exception as e:
        print(f"❌ Exception during Vertex test: {e}")

if __name__ == "__main__":
    main()