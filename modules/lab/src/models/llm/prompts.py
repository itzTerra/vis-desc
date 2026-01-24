from dataclasses import dataclass


SCHEMAS = {
    "base": {
        "type": "object",
        "properties": {
            "rating": {"type": "integer", "minimum": 0, "maximum": 5},
        },
        "required": ["rating"],
        "additionalProperties": False,
    },
    "cot": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "rating": {"type": "integer", "minimum": 0, "maximum": 5},
        },
        "required": ["reasoning", "rating"],
        "additionalProperties": False,
    },
}


PROMPT_PARTS = {
    "system": "You are an expert annotator tasked with rating text segments based on their visual descriptiveness. Your goal is to determine how well a reader can visualize the content described in the text.",
    "task_descriptions": {
        "small": "On a scale of 0-5, rate how well this text allows a reader to visualize the scene, objects, or characters described: 0=not visualizable, 1=simple object/person, 2=recognizable place/basic character, 3=basic scene/detailed face, 4=full scene with details, 5=vivid complete scene.",
        "medium": """Rate this text segment's visual descriptiveness from 0-5 by answering these questions in order:

1. Can you visualize a complete scene with vivid background, foreground objects, AND compelling details (lighting, time of day, atmosphere)? → Rating 5

2. Can you visualize EITHER:
   - A full scene with 3+ of these: background, action, describable characters, distinct visual details? OR
   - A full character with clothing and body details?
   → Rating 4

3. Can you visualize EITHER:
   - A basic scene (e.g., "car on busy street")? OR
   - A detailed face description?
   → Rating 3

4. Can you visualize EITHER:
   - A known building/room/place OR unusual object in detail? OR
   - A character mentioned without details?
   → Rating 2

5. Can you visualize a simple object, animal, body part, clothing item, OR a person physically described in the scene? → Rating 1

6. If none apply → Rating 0

## Important rules:
- Choose the highest-rated visual perspective if multiple exist
- Only rate visual metaphors, not abstract ones
- Don't rate commands, intentions, or hypothetical future actions""",
        "full": '''Rate each text segment on a scale from 0-5, where higher numbers indicate greater visual descriptiveness. Select the **highest applicable rating** found in the segment.

## Rating Scale

**Rating 5** - Full vivid scene
- Can visualize a complete scene with vivid background, foreground objects, and compelling details (e.g., time of day, lighting, atmospheric conditions)
- Example: "The sun was setting behind the mountains, casting long shadows over the bustling village square where children played and vendors packed up their stalls."

**Rating 4** - Full scene with details OR detailed character
- **Option A (Scene)**: Can visualize a full scene with multiple elements - must have at least 3 of:
  1. Background/surroundings
  2. Action/main scene
  3. Describable character(s)
  4. Other visual details (each distinct detail counts separately)
  - Example: "A river flowed through the meadow, with a few trees scattered along its banks and a small wooden bridge crossing over."
- **Option B (Character)**: Can visualize a full character with clothing AND body details
  - Example: "She wore a long red coat over a white dress, her curly hair tied back with a blue ribbon, and her boots were splattered with mud."

**Rating 3** - Basic scene OR detailed face
- **Option A (Scene)**: Can visualize a scene without additional details
  - Examples: "car driving down a busy street" / "the teacher sat down in his chair"
- **Option B (Face)**: Can visualize a detailed face
  - Example: "Brown patches of skin had appeared on his cheeks. The patches extended down the sides of his face. Everything about him was old, including his eyes. They were the same color as the sea and they looked cheerful and untwisted."

**Rating 2** - Recognizable place/object OR basic character
- **Option A (Place/Object)**: Can visualize a commonly-known building, room, place, OR an interesting/unusual object, body part, clothing, or accessory in detail
  - Examples: "castle" / "bathroom" / "field" / "king's sword covered in blood" / "crown with a missing jewel"
- **Option B (Character)**: Can visualize a character without details
  - Example: "king"

**Rating 1** - Simple visual element OR mentioned person
- **Option A (Object/Animal)**: Can visualize an ordinary animal, object, body part, or clothing
  - Examples: "frog" / "book" / "his hands were dirty"
- **Option B (Person)**: Can visualize an unknown-looking person if physically described in the scene by the narrator or another character
  - Example: "Mr. and Mrs. Smith just arrived"
  - Note: Person behind monologue/dialogue doesn't count; theoretical person or person only mentioned in dialogue doesn't count

**Rating 0** - Not visualizable
- Hard to visualize anything concrete

**IMPORTANT**: Add +1 to this rating if objects are described in a **visualizable action** (NOT just hand movement or talking). For example, "a book" = rating 1, but "closing a book" = rating 2 (1 + 1). Maximum rating remains 5.

## Important Rules

### Note 1: Multiple Perspectives
If the text contains multiple visual perspectives, rate **only the perspective with the highest visual value**. Include only details from that chosen perspective, not from others. Example:
- 'Billy climbed the tree wearing his backpack, which contained his slingshot, some pebbles and a magnifying glass' -> PERSPECTIVE 1: 'Billy climbed the tree wearing his backpack' | PERSPECTIVE 2: 'backpack, which contained his slingshot, some pebbles and a magnifying glass'

### Note 2: Metaphorical Content
You can rate metaphorical content **only if** the meaning behind the metaphor is visual. Example:
- ✓ Rateable: "the cathedral advanced like a headless creature in a long, shapeless coat"
- ✗ Not rateable: "he fought his illness as he would have fought a devil"

### Note 3: Theoretical or Spoken-About Content
You can rate theoretical or spoken-about content **unless** it is either: a) Present tense imperative, OR b) Expression of character's purpose, intention, or ability to do something. Example:
- ✓ Rateable: "No", hesitated Jim. "The ranch won't hold no more cattle than it always has."
- ✗ Not rateable: "Take a hold of your note-book, Shawn."
- ✗ Not rateable: "I came back the same way on the chance of finding the moose."''',
    },
    "examples": {
        "base": """## Examples
Here are examples demonstrating each rating level:

<example_1>
<text_segment>If not, please say so. The alternative is in the hands of the Crown. I am the Crown. The Crown may at any time confiscate property and banish malcontents and disturbers.</text_segment>
{
  "rating": 0
}
</example_1>

<example_2>
<text_segment>By this time the fresh new fool's cap made from yesterday's "Bee" would have been pinned on her head. "Quelle injustice! Quelle injustice!" This last apostrophe in a high, whining nasal voice, always procured Pupasse's elevation on the tall three-legged stool in the corner.</text_segment>
{
  "rating": 1
}
</example_2>

<example_3>
<text_segment>So he looked through the deep-blue luminous window once more, beyond the battles we know. From this he turned back shuddering. Again he saw the Professor smile with his lips, though whether at his own weakness, or whether with cynical mirth at the fate of the world, Rodriguez could not say.</text_segment>
{
  "rating": 2
}
</example_3>

<example_4>
<text_segment>When the meal had been arranged the prisoners saw that there was enough for four people instead of two; and the soldiers realized this also, for they turned puzzled looks first on the tables and then on the prisoners. Then they shook all their twin heads gravely and went away, locking the twin doors behind them.</text_segment>
{
  "rating": 3
}
</example_4>

<example_5>
<text_segment>The wonderful wheat-field was cut into a thousand squares and angles and lanes and curves. The big whirring combines passed one another, stopped and waited and turned out of the way, leaving everywhere little patches and cubes of standing wheat, that soon fell before the onslaught of the smaller combines.</text_segment>
{
  "rating": 4
}
</example_5>

<example_6>
<text_segment>Or one might go out into the harbour late upon a winter afternoon, and see it as a city of a million lights, rising like an incantation from the sea. Round about it was an unbroken ring of docks, with ferry-boats and tugs darting everywhere, and vessels which had come from every port in the world, emptying their cargoes into the huge maw of the Metropolis.</text_segment>
{
  "rating": 5
}
</example_6>""",
        "cot": """## Examples
Here are examples demonstrating each rating level:

<example_1>
<text_segment>If not, please say so. The alternative is in the hands of the Crown. I am the Crown. The Crown may at any time confiscate property and banish malcontents and disturbers.</text_segment>
{
  "reasoning": "Part of dialogue about abstract legal/political concepts with no visual elements to visualize.",
  "rating": 0
}
</example_1>

<example_2>
<text_segment>By this time the fresh new fool's cap made from yesterday's "Bee" would have been pinned on her head. "Quelle injustice! Quelle injustice!" This last apostrophe in a high, whining nasal voice, always procured Pupasse's elevation on the tall three-legged stool in the corner.</text_segment>
{
  "reasoning": "Simple objects mentioned (fool's cap, stool) but minimal scene context or spatial detail.",
  "rating": 1
}
</example_2>

<example_3>
<text_segment>So he looked through the deep-blue luminous window once more, beyond the battles we know. From this he turned back shuddering. Again he saw the Professor smile with his lips, though whether at his own weakness, or whether with cynical mirth at the fate of the world, Rodriguez could not say.</text_segment>
{
  "reasoning": "Detailed interesting object (deep-blue luminous window) with some action (unknown person looking through) but limited scene composition.",
  "rating": 2
}
</example_3>

<example_4>
<text_segment>When the meal had been arranged the prisoners saw that there was enough for four people instead of two; and the soldiers realized this also, for they turned puzzled looks first on the tables and then on the prisoners. Then they shook all their twin heads gravely and went away, locking the twin doors behind them.</text_segment>
{
  "reasoning": "Basic scene with multiple characters, actions, and spatial relationships (tables, doors, people looking at each other).",
  "rating": 3
}
</example_4>

<example_5>
<text_segment>The wonderful wheat-field was cut into a thousand squares and angles and lanes and curves. The big whirring combines passed one another, stopped and waited and turned out of the way, leaving everywhere little patches and cubes of standing wheat, that soon fell before the onslaught of the smaller combines.</text_segment>
{
  "reasoning": "Full scene with detailed spatial arrangement, movement, and specific visual patterns (geometric shapes, machinery interactions).",
  "rating": 4
}
</example_5>

<example_6>
<text_segment>Or one might go out into the harbour late upon a winter afternoon, and see it as a city of a million lights, rising like an incantation from the sea. Round about it was an unbroken ring of docks, with ferry-boats and tugs darting everywhere, and vessels which had come from every port in the world, emptying their cargoes into the huge maw of the Metropolis.</text_segment>
{
  "reasoning": "Vivid, complete scene with rich atmospheric details (time of day, season, lighting), comprehensive spatial layout, dynamic movement, and evocative imagery.",
  "rating": 5
}
</example_6>""",
    },
    "output_format": {
        "base": """## Output Format
Provide your answer as a JSON object. You must not answer with anything other than the JSON. Your message must end with the JSON closing curly bracket. You must follow this format:
{
  "rating": <integer 0-5>
}""",
        "cot": """## Output Format
Provide your answer as a JSON object. You must not answer with anything other than the JSON. Your message must end with the JSON closing curly bracket. You must follow this format:
{
  "reasoning": "<brief explanation of your rating decision>",
  "rating": <integer 0-5>
}""",
    },
    "input": """## Input
Rate the following text segment:

<text_segment>{{TEXT_SEGMENT}}</text_segment>""",
}


@dataclass
class Prompt:
    system: str
    guideline: str
    examples: str | None
    suffix: str
    suffix_key: str

    def build_user_prompt(self, text_segment: str) -> str:
        """Build the complete user prompt with text segment."""
        user_prompt = self.guideline
        if self.examples:
            user_prompt += f"\n\n{self.examples}"
        user_prompt += f"\n\n{self.suffix.replace('{{TEXT_SEGMENT}}', text_segment)}"
        return user_prompt

    def get_id(self) -> str:
        """Build a unique identifier for the prompt configuration."""
        example_part = (
            ("few-shot-cot" if "reasoning" in self.examples else "few-shot")
            if self.examples
            else "zero-shot"
        )
        cot_part = "cot" if "reasoning" in self.suffix else "no-cot"
        return f"{example_part}_{cot_part}_{self.guideline[:10].replace(' ', '-')}"


COMBINATION_PLANS = [
    {
        "task_descriptions": ("small",),
        "include_examples": False,
        "cot_options": ("none", "zero-shot"),
    },
    {
        "task_descriptions": ("small",),
        "include_examples": True,
        "cot_options": ("none", "zero-shot"),
    },
    {
        "task_descriptions": ("medium",),
        "include_examples": False,
        "cot_options": ("none", "zero-shot"),
    },
    {
        "task_descriptions": ("medium",),
        "include_examples": True,
        "cot_options": ("none", "zero-shot", "few-shot"),
    },
    {
        "task_descriptions": ("full",),
        "include_examples": False,
        "cot_options": ("none", "zero-shot"),
    },
    {
        "task_descriptions": ("full",),
        "include_examples": True,
        "cot_options": ("none", "zero-shot", "few-shot"),
    },
]


def select_examples(include_examples: bool, cot: str) -> str | None:
    if not include_examples:
        return None
    if cot == "few-shot":
        return PROMPT_PARTS["examples"]["cot"]
    return PROMPT_PARTS["examples"]["base"]


def select_suffix(cot: str) -> str:
    key = "cot" if cot != "none" else "base"
    suffix = PROMPT_PARTS["output_format"][key] + "\n\n" + PROMPT_PARTS["input"]
    return key, suffix


def schema_for_suffix_key(key: str) -> dict:
    return SCHEMAS[key]


def schema_for_prompt(prompt: Prompt) -> dict:
    return schema_for_suffix_key(prompt.suffix_key)


PROMPTS: list[Prompt] = []
for plan in COMBINATION_PLANS:
    for task_descr_key in plan["task_descriptions"]:
        for cot_option in plan["cot_options"]:
            PROMPTS.append(
                Prompt(
                    system=PROMPT_PARTS["system"],
                    guideline=PROMPT_PARTS["task_descriptions"][task_descr_key],
                    examples=select_examples(plan["include_examples"], cot_option),
                    suffix_key=select_suffix(cot_option)[0],
                    suffix=select_suffix(cot_option)[1],
                )
            )

assert len(PROMPTS) == 14

OPTIMIZED_PARTS = [
    # 0.962 MSE
    {
        "system": "You are an expert annotator tasked with rating text segments based on their visual descriptiveness. Your goal is to determine how well a reader can visualize the content described in the text.",
        "guideline": 'Rate each text segment on a scale from 0-5, where higher numbers indicate greater visual descriptiveness. Select the **highest applicable rating** found in the segment.\n\n## Rating Scale\n\n**Rating 5** - Full vivid scene\n- Can visualize a complete scene with vivid background, foreground objects, and compelling details (e.g., time of day, lighting, atmospheric conditions)\n- Example: "The sun was setting behind the mountains, casting long shadows over the bustling village square where children played and vendors packed up their stalls."\n\n**Rating 4** - Full scene with details OR detailed character\n- **Option A (Scene)**: Can visualize a full scene with multiple elements - must have at least 3 of:\n  1. Background/surroundings\n  2. Action/main scene\n  3. Describable character(s)\n  4. Other visual details (each distinct detail counts separately)\n  - Example: "A river flowed through the meadow, with a few trees scattered along its banks and a small wooden bridge crossing over."\n- **Option B (Character)**: Can visualize a full character with clothing AND body details\n  - Example: "She wore a long red coat over a white dress, her curly hair tied back with a blue ribbon, and her boots were splattered with mud."\n\n**Rating 3** - Basic scene OR detailed face\n- **Option A (Scene)**: Can visualize a scene without additional details\n  - Examples: "car driving down a busy street" / "the teacher sat down in his chair"\n- **Option B (Face)**: Can visualize a detailed face\n  - Example: "Brown patches of skin had appeared on his cheeks. The patches extended down the sides of his face. Everything about him was old, including his eyes. They were the same color as the sea and they looked cheerful and untwisted."\n\n**Rating 2** - Recognizable place/object OR basic character\n- **Option A (Place/Object)**: Can visualize a commonly-known building, room, place, OR an interesting/unusual object, body part, clothing, or accessory in detail\n  - Examples: "castle" / "bathroom" / "field" / "king\'s sword covered in blood" / "crown with a missing jewel"\n- **Option B (Character)**: Can visualize a character without details\n  - Example: "king"\n\n**Rating 1** - Simple visual element OR mentioned person\n- **Option A (Object/Animal)**: Can visualize an ordinary animal, object, body part, or clothing\n  - Examples: "frog" / "book" / "his hands were dirty"\n- **Option B (Person)**: Can visualize an unknown-looking person if physically described in the scene by the narrator or another character\n  - Example: "Mr. and Mrs. Smith just arrived"\n  - Note: Person behind monologue/dialogue doesn\'t count; theoretical person or person only mentioned in dialogue doesn\'t count\n\n**Rating 0** - Not visualizable\n- Hard to visualize anything concrete\n\n**IMPORTANT**: Add +1 to this rating if objects are described in a **visualizable action** (NOT just hand movement or talking). For example, "a book" = rating 1, but "closing a book" = rating 2 (1 + 1). Maximum rating remains 5.\n\n## Important Rules\n\n### Note 1: Multiple Perspectives\nIf the text contains multiple visual perspectives, rate **only the perspective with the highest visual value**. Include only details from that chosen perspective, not from others. Example:\n- \'Billy climbed the tree wearing his backpack, which contained his slingshot, some pebbles and a magnifying glass\' -> PERSPECTIVE 1: \'Billy climbed the tree wearing his backpack\' | PERSPECTIVE 2: \'backpack, which contained his slingshot, some pebbles and a magnifying glass\'\n\n### Note 2: Metaphorical Content\nYou can rate metaphorical content **only if** the meaning behind the metaphor is visual. Example:\n- ✓ Rateable: "the cathedral advanced like a headless creature in a long, shapeless coat"\n- ✗ Not rateable: "he fought his illness as he would have fought a devil"\n\n### Note 3: Theoretical or Spoken-About Content\nYou can rate theoretical or spoken-about content **unless** it is either: a) Present tense imperative, OR b) Expression of character\'s purpose, intention, or ability to do something. Example:\n- ✓ Rateable: "No", hesitated Jim. "The ranch won\'t hold no more cattle than it always has."\n- ✗ Not rateable: "Take a hold of your note-book, Shawn."\n- ✗ Not rateable: "I came back the same way on the chance of finding the moose."',
        "examples": '## Examples\nHere are examples demonstrating each rating level:\n\n<example_1>\n<text_segment>If not, please say so. The alternative is in the hands of the Crown. I am the Crown. The Crown may at any time confiscate property and banish malcontents and disturbers.</text_segment>\n{\n  "rating": 0\n}\n</example_1>\n\n<example_2>\n<text_segment>By this time the fresh new fool\'s cap made from yesterday\'s "Bee" would have been pinned on her head. "Quelle injustice! Quelle injustice!" This last apostrophe in a high, whining nasal voice, always procured Pupasse\'s elevation on the tall three-legged stool in the corner.</text_segment>\n{\n  "rating": 1\n}\n</example_2>\n\n<example_3>\n<text_segment>So he looked through the deep-blue luminous window once more, beyond the battles we know. From this he turned back shuddering. Again he saw the Professor smile with his lips, though whether at his own weakness, or whether with cynical mirth at the fate of the world, Rodriguez could not say.</text_segment>\n{\n  "rating": 2\n}\n</example_3>\n\n<example_4>\n<text_segment>When the meal had been arranged the prisoners saw that there was enough for four people instead of two; and the soldiers realized this also, for they turned puzzled looks first on the tables and then on the prisoners. Then they shook all their twin heads gravely and went away, locking the twin doors behind them.</text_segment>\n{\n  "rating": 3\n}\n</example_4>\n\n<example_5>\n<text_segment>The wonderful wheat-field was cut into a thousand squares and angles and lanes and curves. The big whirring combines passed one another, stopped and waited and turned out of the way, leaving everywhere little patches and cubes of standing wheat, that soon fell before the onslaught of the smaller combines.</text_segment>\n{\n  "rating": 4\n}\n</example_5>\n\n<example_6>\n<text_segment>Or one might go out into the harbour late upon a winter afternoon, and see it as a city of a million lights, rising like an incantation from the sea. Round about it was an unbroken ring of docks, with ferry-boats and tugs darting everywhere, and vessels which had come from every port in the world, emptying their cargoes into the huge maw of the Metropolis.</text_segment>\n{\n  "rating": 5\n}\n</example_6>',
        "output_format": '{\n  "rating": 0\n}',
        "input": "## Input\nRate the following text segment:\n\n<text_segment>{{TEXT_SEGMENT}}</text_segment>",
    },
    # 1.077 ARMSE
    {
        "system": "You are an expert annotator tasked with rating text segments based on their visual descriptiveness. Your goal is to determine how well a reader can visualize the content described in the text.",
        "guideline": 'Rate each text segment on a scale from 0-5, where higher numbers indicate greater visual descriptiveness. Select the **highest applicable rating** found in the segment.\n\n## Rating Scale\n\n**Rating 5** - Full vivid scene\n- Can visualize a complete scene with vivid background, foreground objects, and compelling details (e.g., time of day, lighting, atmospheric conditions)\n- Example: "The sun was setting behind the mountains, casting long shadows over the bustling village square where children played and vendors packed up their stalls."\n\n**Rating 4** - Full scene with details OR detailed character\n- **Option A (Scene)**: Can visualize a full scene with multiple elements - must have at least 3 of:\n  1. Background/surroundings\n  2. Action/main scene\n  3. Describable character(s)\n  4. Other visual details (each distinct detail counts separately)\n  - Example: "A river flowed through the meadow, with a few trees scattered along its banks and a small wooden bridge crossing over."\n- **Option B (Character)**: Can visualize a full character with clothing AND body details\n  - Example: "She wore a long red coat over a white dress, her curly hair tied back with a blue ribbon, and her boots were splattered with mud."\n\n**Rating 3** - Basic scene OR detailed face\n- **Option A (Scene)**: Can visualize a scene without additional details\n  - Examples: "car driving down a busy street" / "the teacher sat down in his chair"\n- **Option B (Face)**: Can visualize a detailed face\n  - Example: "Brown patches of skin had appeared on his cheeks. The patches extended down the sides of his face. Everything about him was old, including his eyes. They were the same color as the sea and they looked cheerful and untwisted."\n\n**Rating 2** - Recognizable place/object OR basic character\n- **Option A (Place/Object)**: Can visualize a commonly-known building, room, place, OR an interesting/unusual object, body part, clothing, or accessory in detail\n  - Examples: "castle" / "bathroom" / "field" / "king\'s sword covered in blood" / "crown with a missing jewel"\n- **Option B (Character)**: Can visualize a character without details\n  - Example: "king"\n\n**Rating 1** - Simple visual element OR mentioned person\n- **Option A (Object/Animal)**: Can visualize an ordinary animal, object, body part, or clothing\n  - Examples: "frog" / "book" / "his hands were dirty"\n- **Option B (Person)**: Can visualize an unknown-looking person if physically described in the scene by the narrator or another character\n  - Example: "Mr. and Mrs. Smith just arrived"\n  - Note: Person behind monologue/dialogue doesn\'t count; theoretical person or person only mentioned in dialogue doesn\'t count\n\n**Rating 0** - Not visualizable\n- Hard to visualize anything concrete\n\n**IMPORTANT**: Add +1 to this rating if objects are described in a **visualizable action** (NOT just hand movement or talking). For example, "a book" = rating 1, but "closing a book" = rating 2 (1 + 1). Maximum rating remains 5.\n\n## Important Rules\n\n### Note 1: Multiple Perspectives\nIf the text contains multiple visual perspectives, rate **only the perspective with the highest visual value**. Include only details from that chosen perspective, not from others. Example:\n- \'Billy climbed the tree wearing his backpack, which contained his slingshot, some pebbles and a magnifying glass\' -> PERSPECTIVE 1: \'Billy climbed the tree wearing his backpack\' | PERSPECTIVE 2: \'backpack, which contained his slingshot, some pebbles and a magnifying glass\'\n\n### Note 2: Metaphorical Content\nYou can rate metaphorical content **only if** the meaning behind the metaphor is visual. Example:\n- ✓ Rateable: "the cathedral advanced like a headless creature in a long, shapeless coat"\n- ✗ Not rateable: "he fought his illness as he would have fought a devil"\n\n### Note 3: Theoretical or Spoken-About Content\nYou can rate theoretical or spoken-about content **unless** it is either: a) Present tense imperative, OR b) Expression of character\'s purpose, intention, or ability to do something. Example:\n- ✓ Rateable: "No", hesitated Jim. "The ranch won\'t hold no more cattle than it always has."\n- ✗ Not rateable: "Take a hold of your note-book, Shawn."\n- ✗ Not rateable: "I came back the same way on the chance of finding the moose."',
        "examples": '## Examples\nHere are examples demonstrating each rating level:\n\n<example_1>\n<text_segment>If not, please say so. The alternative is in the hands of the Crown. I am the Crown. The Crown may at any time confiscate property and banish malcontents and disturbers.</text_segment>\n{\n  "rating": 0\n}\n</example_1>\n\n<example_2>\n<text_segment>By this time the fresh new fool\'s cap made from yesterday\'s "Bee" would have been pinned on her head. "Quelle injustice! Quelle injustice!" This last apostrophe in a high, whining nasal voice, always procured Pupasse\'s elevation on the tall three-legged stool in the corner.</text_segment>\n{\n  "rating": 1\n}\n</example_2>\n\n<example_3>\n<text_segment>So he looked through the deep-blue luminous window once more, beyond the battles we know. From this he turned back shuddering. Again he saw the Professor smile with his lips, though whether at his own weakness, or whether with cynical mirth at the fate of the world, Rodriguez could not say.</text_segment>\n{\n  "rating": 2\n}\n</example_3>\n\n<example_4>\n<text_segment>When the meal had been arranged the prisoners saw that there was enough for four people instead of two; and the soldiers realized this also, for they turned puzzled looks first on the tables and then on the prisoners. Then they shook all their twin heads gravely and went away, locking the twin doors behind them.</text_segment>\n{\n  "rating": 3\n}\n</example_4>\n\n<example_5>\n<text_segment>The wonderful wheat-field was cut into a thousand squares and angles and lanes and curves. The big whirring combines passed one another, stopped and waited and turned out of the way, leaving everywhere little patches and cubes of standing wheat, that soon fell before the onslaught of the smaller combines.</text_segment>\n{\n  "rating": 4\n}\n</example_5>\n\n<example_6>\n<text_segment>Or one might go out into the harbour late upon a winter afternoon, and see it as a city of a million lights, rising like an incantation from the sea. Round about it was an unbroken ring of docks, with ferry-boats and tugs darting everywhere, and vessels which had come from every port in the world, emptying their cargoes into the huge maw of the Metropolis.</text_segment>\n{\n  "rating": 5\n}\n</example_6>',
        "output_format": '{ \n  "rating": 0 \n}',
        "input": "## Input\nRate the following text segment:\n\n<text_segment>{{TEXT_SEGMENT}}</text_segment>",
    },
]
