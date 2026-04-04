As a visual perception analyst, assess the degree to which a given text segment activates the mind's eye, quantifying the clarity and detail with which the content can be mentally visualized. Provide a precise evaluation of the vividness of the mental imagery evoked by the text, focusing on the specificity and intensity of the visual elements it conjures.

On a scale of 0-5, rate how well this text allows a reader to visualize the scene, objects, or characters described: 0=not visualizable, 1=simple object/person, 2=recognizable place/basic character, 3=basic scene/detailed face, 4=full scene with details, 5=vivid complete scene.

## Examples
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
</example_6>

## Output Format
Provide your answer as a JSON object. You must not answer with anything other than the JSON. Your message must end with the JSON closing curly bracket. You must follow this format:
{
      "rating": <integer 0-5>,
  "confidence": <integer 0-5>,
  "justification": "<one-sentence justification>"
}

Read the text segment once to get an overall impression. Then, scan it for specific visual markers that may influence your rating. After completing these steps, assign a rating to the text segment.

<text_segment>{{TEXT_SEGMENT}}</text_segment>
