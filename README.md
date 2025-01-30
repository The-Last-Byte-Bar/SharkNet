# SharkNet
🦈 SharkNet: A collaborative dataset for LLM training, swimming through the depths of human knowledge. Community-driven Q&amp;A pairs for fine-tuning language models.

# SharkNet-Training

A collaborative repository for collecting high-quality question-answer pairs focused on ErgoScript and Scala-like languages for LLM fine-tuning. Our goal is to create a specialized training dataset for generating ErgoScript code for the Ergo blockchain.

## Purpose

This repository aims to build a comprehensive dataset of ErgoScript and blockchain-related Q&A pairs to fine-tune Large Language Models. The focus is on smart contract development, blockchain interactions, and Ergo-specific implementations.

## Example Q&A Format

Here's an example of the type of Q&A pairs we're looking for:

```json
{
  "conversations": [
    {
      "id": "ergo_001",
      "question": "How do I write an ErgoScript contract that locks funds until a specific height?",
      "answer": "Here's how to create a height-locked contract in ErgoScript with detailed explanation:\n\n```scala\n{
    // HEIGHT is a blockchain context variable
    // Input parameters
    val deadlineHeight = 100000L // Example height
    
    // Core logic
    HEIGHT >= deadlineHeight
}\n```\n\nThis contract:\n1. Uses the `HEIGHT` context variable to access current block height\n2. Compares it against a specified deadline\n3. Only allows spending when the current height is >= deadline\n\nYou can deploy this contract by wrapping it in a proper Ergo transaction output.",
      "metadata": {
        "domain": "smart_contracts",
        "difficulty": "beginner",
        "tags": ["ergoscript", "time-lock", "blockchain-height"],
        "source": "original",
        "language": "en",
        "contributor": "shark_coder"
      }
    }
  ]
}
```

## Another Complex Example

```json
{
  "conversations": [
    {
      "id": "ergo_002",
      "question": "How do I create an ErgoScript contract for a simple atomic swap between ERG and a token?",
      "answer": "Here's an ErgoScript contract implementing an atomic swap with explanation:\n\n```scala\n{
    // Contract Parameters
    val ergAmount = 1000000000L // 1 ERG in nanoERG
    val tokenId = fromBase58(\"token_id_here\")
    val tokenAmount = 100L
    
    // Ensure the output box has the expected ERG amount
    val properERGAmount = OUTPUTS(0).value >= ergAmount
    
    // Verify token transfer
    val properTokenTransfer = OUTPUTS(0).tokens.exists({ (token: (Coll[Byte], Long)) =>
        token._1 == tokenId && token._2 >= tokenAmount
    })
    
    // Combined conditions
    properERGAmount && properTokenTransfer
}\n```\n\nThis contract:\n1. Defines expected amounts for both ERG and tokens\n2. Checks if output box contains minimum ERG amount\n3. Verifies proper token transfer using token ID\n4. Combines conditions with logical AND\n\nTo use:\n1. Deploy as contract address\n2. Send ERG to contract address\n3. Anyone can fulfill by sending required tokens",
      "metadata": {
        "domain": "smart_contracts",
        "difficulty": "intermediate",
        "tags": ["ergoscript", "atomic-swap", "tokens"],
        "source": "original",
        "language": "en",
        "contributor": "shark_coder"
      }
    }
  ]
}
```

## How to Contribute

### File Structure
Place your contributions in the `data` directory using the following format:
```
data/
  ├── ergoscript/
      ├── smart_contracts/
      ├── dapps/
      ├── oracles/
      └── tokens/
  ├── scala_concepts/
      ├── basics/
      ├── advanced/
      └── patterns/
```

### Quality Guidelines for ErgoScript Content

1. **Correctness**: Code must be syntactically correct and follow ErgoScript best practices
2. **Completeness**: Include all necessary imports and context variables
3. **Security**: Highlight security considerations for smart contracts
4. **Documentation**: Explain key concepts and potential edge cases
5. **Testing**: Include test scenarios where applicable
6. **Gas Efficiency**: Note any relevant optimization considerations

### Specific Topic Areas to Cover

1. Basic ErgoScript Concepts
   - Syntax and data types
   - Context variables
   - Box operations

2. Smart Contract Patterns
   - Time-locked contracts
   - Multi-signature schemes
   - Oracle interactions

3. Token Operations
   - Minting
   - Burning
   - Transfer logic

4. DApp Integration
   - Frontend interaction
   - Transaction building
   - Wallet integration

## Tools and Scripts

The `tools` directory contains helpful utilities:
- `validate_ergo.py`: Validates ErgoScript syntax and common patterns
- `convert_to_training.py`: Prepares data for LLM training
- `test_contracts.py`: Basic testing framework for contracts

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Important Considerations

- **Security**: Review all smart contract code thoroughly
- **Testing**: Include test cases for different scenarios
- **Gas Optimization**: Consider execution costs
- **Documentation**: Explain complex patterns clearly

## Getting Started

1. Clone the repository
2. Set up the Python environment using `requirements.txt`
3. Review example ErgoScript Q&A pairs in `data/examples/`
4. Use the validation tool: `python tools/validate_ergo.py your_file.json`

## Contact

For questions or concerns, please open an issue or contact the maintainers.
