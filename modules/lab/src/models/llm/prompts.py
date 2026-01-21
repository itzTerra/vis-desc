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
    "action_bonus": {
        "type": "object",
        "properties": {
            "action_bonus_applied": {"type": "boolean"},
            "rating_without_action_bonus": {
                "type": "integer",
                "minimum": 0,
                "maximum": 5,
            },
        },
        "required": ["action_bonus_applied", "rating_without_action_bonus"],
        "additionalProperties": False,
    },
    "zs_cot": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "rating": {"type": "integer", "minimum": 0, "maximum": 5},
        },
        "required": ["reasoning", "rating"],
        "additionalProperties": False,
    },
    "cot_action_bonus": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "action_bonus_applied": {"type": "boolean"},
            "rating_without_action_bonus": {
                "type": "integer",
                "minimum": 0,
                "maximum": 5,
            },
        },
        "required": [
            "reasoning",
            "action_bonus_applied",
            "rating_without_action_bonus",
        ],
        "additionalProperties": False,
    },
}


PROMPT_PARTS = {
    "system": "You are an expert annotator tasked with rating text segments based on their visual descriptiveness. Your goal is to determine how well a reader can visualize the content described in the text.",
    "guidelines": {
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

## Action Bonus Consideration

In some cases (usually ratings 1, 2), the objects may be described in a **visualizable** action (**NOT** just hand movement or talking - those are not visualizable). For example, "a book" would be rating 1, but "closing a book" would receive the action bonus. Determine whether this applies to the text segment.

## Important Rules

### Note 1: Multiple Perspectives
If the text contains multiple visual perspectives, rate **only the perspective with the highest visual value**. Include only details from that chosen perspective, not from others.

**Example:**
- Text: "Billy climbed the tree wearing his backpack, which contained his slingshot, some pebbles and a magnifying glass"
- **Perspective 1**: "Billy climbed the tree wearing his backpack"
- **Perspective 2**: "backpack, which contained his slingshot, some pebbles and a magnifying glass"
- Rate the higher-value perspective only

### Note 2: Metaphorical Content
You can rate metaphorical content **only if** the meaning behind the metaphor is visual.

- ✓ Rateable: "the cathedral advanced like a headless creature in a long, shapeless coat"
- ✗ Not rateable: "he fought his illness as he would have fought a devil"

### Note 3: Theoretical or Spoken-About Content
You can rate theoretical or spoken-about content **unless** it is either:
- a) Present tense imperative, OR
- b) Expression of character's purpose, intention, or ability to do something

**Examples:**
- ✓ Rateable: "No", hesitated Jim. "The ranch won't hold no more cattle than it always has."
- ✗ Not rateable: "Take a hold of your note-book, Shawn."
- ✗ Not rateable: "I came back the same way on the chance of finding the moose."''',
    },
    "examples": {
        "base": """## Examples
Here are examples demonstrating each rating level:

<example>
<text_segment>If not, please say so. The alternative is in the hands of the Crown. I am the Crown. The Crown may at any time confiscate property and banish malcontents and disturbers.</text_segment>
<rating>0</rating>
</example>

<example>
<text_segment>By this time the fresh new fool's cap made from yesterday's "Bee" would have been pinned on her head. "Quelle injustice! Quelle injustice!" This last apostrophe in a high, whining nasal voice, always procured Pupasse's elevation on the tall three-legged stool in the corner.</text_segment>
<rating>1</rating>
</example>

<example>
<text_segment>So he looked through the deep-blue luminous window once more, beyond the battles we know. From this he turned back shuddering. Again he saw the Professor smile with his lips, though whether at his own weakness, or whether with cynical mirth at the fate of the world, Rodriguez could not say.</text_segment>
<rating>2</rating>
</example>

<example>
<text_segment>When the meal had been arranged the prisoners saw that there was enough for four people instead of two; and the soldiers realized this also, for they turned puzzled looks first on the tables and then on the prisoners. Then they shook all their twin heads gravely and went away, locking the twin doors behind them.</text_segment>
<rating>3</rating>
</example>

<example>
<text_segment>The wonderful wheat-field was cut into a thousand squares and angles and lanes and curves. The big whirring combines passed one another, stopped and waited and turned out of the way, leaving everywhere little patches and cubes of standing wheat, that soon fell before the onslaught of the smaller combines.</text_segment>
<rating>4</rating>
</example>

<example>
<text_segment>Or one might go out into the harbour late upon a winter afternoon, and see it as a city of a million lights, rising like an incantation from the sea. Round about it was an unbroken ring of docks, with ferry-boats and tugs darting everywhere, and vessels which had come from every port in the world, emptying their cargoes into the huge maw of the Metropolis.</text_segment>
<rating>5</rating>
</example>""",
        "cot": """## Examples
Here are examples demonstrating each rating level:

<example>
<text_segment>If not, please say so. The alternative is in the hands of the Crown. I am the Crown. The Crown may at any time confiscate property and banish malcontents and disturbers.</text_segment>
<reasoning>Part of dialogue about abstract legal/political concepts with no visual elements to visualize.</reasoning>
<rating>0</rating>
</example>

<example>
<text_segment>By this time the fresh new fool's cap made from yesterday's "Bee" would have been pinned on her head. "Quelle injustice! Quelle injustice!" This last apostrophe in a high, whining nasal voice, always procured Pupasse's elevation on the tall three-legged stool in the corner.</text_segment>
<reasoning>Simple objects mentioned (fool's cap, stool) but minimal scene context or spatial detail.</reasoning>
<rating>1</rating>
</example>

<example>
<text_segment>So he looked through the deep-blue luminous window once more, beyond the battles we know. From this he turned back shuddering. Again he saw the Professor smile with his lips, though whether at his own weakness, or whether with cynical mirth at the fate of the world, Rodriguez could not say.</text_segment>
<reasoning>Detailed interesting object (deep-blue luminous window) with some action (unknown person looking through) but limited scene composition.</reasoning>
<rating>2</rating>
</example>

<example>
<text_segment>When the meal had been arranged the prisoners saw that there was enough for four people instead of two; and the soldiers realized this also, for they turned puzzled looks first on the tables and then on the prisoners. Then they shook all their twin heads gravely and went away, locking the twin doors behind them.</text_segment>
<reasoning>Basic scene with multiple characters, actions, and spatial relationships (tables, doors, people looking at each other).</reasoning>
<rating>3</rating>
</example>

<example>
<text_segment>The wonderful wheat-field was cut into a thousand squares and angles and lanes and curves. The big whirring combines passed one another, stopped and waited and turned out of the way, leaving everywhere little patches and cubes of standing wheat, that soon fell before the onslaught of the smaller combines.</text_segment>
<reasoning>Full scene with detailed spatial arrangement, movement, and specific visual patterns (geometric shapes, machinery interactions).</reasoning>
<rating>4</rating>
</example>

<example>
<text_segment>Or one might go out into the harbour late upon a winter afternoon, and see it as a city of a million lights, rising like an incantation from the sea. Round about it was an unbroken ring of docks, with ferry-boats and tugs darting everywhere, and vessels which had come from every port in the world, emptying their cargoes into the huge maw of the Metropolis.</text_segment>
<reasoning>Vivid, complete scene with rich atmospheric details (time of day, season, lighting), comprehensive spatial layout, dynamic movement, and evocative imagery.</reasoning>
<rating>5</rating>
</example>""",
    },
    "suffix": {
        "base": """## Input
Rate the following text segment:

<text_segment>
{{TEXT_SEGMENT}}
</text_segment>

## Output Format
Provide your rating as a JSON object right after the string '---YOUR ANSWER BELOW---':

```json
{
  "rating": <integer 0-5>
}
```

**CRITICAL RULE**: You must not answer with anything other than the JSON. Your message must end with the JSON.

---YOUR ANSWER BELOW---""",
        "action_bonus": """## Input
Rate the following text segment:

<text_segment>
{{TEXT_SEGMENT}}
</text_segment>

## Output Format
Provide your rating as a JSON object right after the string '---YOUR ANSWER BELOW---':

```json
{
  "action_bonus_applied": <boolean>,
  "rating_without_action_bonus": <integer 0-5>
}
```

**CRITICAL RULE**: You must not answer with anything other than the JSON. Your message must end with the JSON.

---YOUR ANSWER BELOW---""",
        "zs_cot": """## Input
Rate the following text segment:

<text_segment>
{{TEXT_SEGMENT}}
</text_segment>

## Output Format
Provide your rating as a JSON object right after the string '---YOUR ANSWER BELOW---':

```json
{
  "reasoning": "<brief explanation of your rating decision>",
  "rating": <integer 0-5>
}
```

**CRITICAL RULE**: You must not answer with anything other than the JSON. Your message must end with the JSON.

---YOUR ANSWER BELOW---""",
        "cot_action_bonus": """## Input
Rate the following text segment:

<text_segment>
{{TEXT_SEGMENT}}
</text_segment>

## Output Format
Provide your rating as a JSON object right after the string '---YOUR ANSWER BELOW---':

```json
{
  "reasoning": "<brief explanation of your rating decision>",
  "action_bonus_applied": <boolean>,
  "rating_without_action_bonus": <integer 0-5>
}
```

**CRITICAL RULE**: You must not answer with anything other than the JSON. Your message must end with the JSON.

---YOUR ANSWER BELOW---""",
    },
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
        action_bonus_part = "action" if "action_bonus" in self.suffix else "no-action"
        cot_part = "cot" if "reasoning" in self.suffix else "no-cot"
        return f"{example_part}_{cot_part}_{action_bonus_part}_{self.guideline[:10].replace(' ', '-')}"


GUIDELINE_CONFIGS = {
    "small": {"text": PROMPT_PARTS["guidelines"]["small"], "action_bonus": False},
    "medium": {"text": PROMPT_PARTS["guidelines"]["medium"], "action_bonus": False},
    "full": {"text": PROMPT_PARTS["guidelines"]["full"], "action_bonus": True},
}


COMBINATION_PLANS = [
    {
        "guidelines": ("small",),
        "include_examples": False,
        "cot_options": ("none", "zero-shot"),
    },
    {
        "guidelines": ("small",),
        "include_examples": True,
        "cot_options": ("none", "zero-shot"),
    },
    {
        "guidelines": ("medium",),
        "include_examples": False,
        "cot_options": ("none", "zero-shot"),
    },
    {
        "guidelines": ("medium",),
        "include_examples": True,
        "cot_options": ("none", "zero-shot", "few-shot"),
    },
    {
        "guidelines": ("full",),
        "include_examples": False,
        "cot_options": ("none", "zero-shot"),
    },
    {
        "guidelines": ("full",),
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


def select_suffix(has_action_bonus: bool, cot: str) -> str:
    if has_action_bonus:
        key = "cot_action_bonus" if cot != "none" else "action_bonus"
    else:
        key = "zs_cot" if cot != "none" else "base"
    return key, PROMPT_PARTS["suffix"][key]


def schema_for_suffix_key(key: str) -> dict:
    return SCHEMAS[key]


def schema_for_prompt(prompt: Prompt) -> dict:
    return schema_for_suffix_key(prompt.suffix_key)


PROMPTS: list[Prompt] = []
for plan in COMBINATION_PLANS:
    for guideline_key in plan["guidelines"]:
        guideline_config = GUIDELINE_CONFIGS[guideline_key]
        for cot_option in plan["cot_options"]:
            PROMPTS.append(
                Prompt(
                    system=PROMPT_PARTS["system"],
                    guideline=guideline_config["text"],
                    examples=select_examples(plan["include_examples"], cot_option),
                    suffix_key=select_suffix(
                        guideline_config["action_bonus"], cot_option
                    )[0],
                    suffix=select_suffix(guideline_config["action_bonus"], cot_option)[
                        1
                    ],
                )
            )

assert len(PROMPTS) == 14
