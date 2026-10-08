// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	site: 'https://lliquid.github.io',
	base: '/agentcore-rl-toolkit',
	integrations: [
		starlight({
			title: 'ART',
			description: 'RL training on top of Bedrock AgentCore Runtime.',
			customCss: ['./src/styles/custom.css'],
			head: [
				{
					tag: 'script',
					attrs: { is: 'inline' },
					content: `
						try {
							if (!localStorage.getItem('starlight-theme')) {
								localStorage.setItem('starlight-theme', 'dark');
							}
							document.documentElement.dataset.theme =
								localStorage.getItem('starlight-theme') || 'dark';
						} catch {
							document.documentElement.dataset.theme = 'dark';
						}
					`,
				},
			],
			social: [
				{
					icon: 'github',
					label: 'GitHub',
					href: 'https://github.com/awslabs/agentcore-rl-toolkit',
				},
			],
			sidebar: [
				{ label: 'Overview', slug: 'guides/overview' },
				{
					label: 'Setup Guide',
					items: [
						{ label: 'Prepare agent for RL', slug: 'guides/agent-adaptation' },
						{
							label: 'Training Integrations',
							items: [
								{ label: 'verl', slug: 'guides/verl-backend-setup' },
								{ label: 'slime', slug: 'guides/slime-backend-setup' },
								{
									label: 'SkyRL ↗',
									link: 'https://github.com/awslabs/agentcore-rl-toolkit/blob/main/src/agentcore_rl_toolkit/backends/tinker_api/README.md',
								},
								{
									label: 'Tinker ↗',
									link: 'https://github.com/awslabs/agentcore-rl-toolkit/blob/main/src/agentcore_rl_toolkit/backends/tinker_api/README.md',
								},
								{
									label: 'rLLM ↗',
									link: 'https://docs.rllm-project.com/agent-runtimes/agentcore',
								},
							],
						},
					],
				},
				{
					label: 'Examples',
					items: [
						{ label: 'Overview', slug: 'examples' },
						{ label: 'Math (GSM8K)', slug: 'examples/strands-math-agent' },
						{ label: 'AppWorld', slug: 'examples/strands-appworld-agent' },
						{ label: 'MigrationBench', slug: 'examples/strands-migration-agent' },
						{ label: 'OfficeBench', slug: 'examples/strands-officebench-agent' },
						{ label: 'SWE-Gym', slug: 'examples/openhands-swegym-agent' },
					],
				},
				{
					label: 'Blog',
					items: [
						{ label: 'Overview', slug: 'blog' },
						{
							label: 'Training multi-step agents with RL',
							slug: 'blog/training-multi-step-agents-rl',
						},
						{
							label: 'Growing verifiable training data',
							slug: 'blog/data-synthesis-v0',
						},
						{
							label: 'Stable Updates, Fewer Sequences',
							slug: 'blog/variable-rows-linear-mode',
						},
						{
							label: 'Async RL performance optimization',
							slug: 'blog/async-rl-performance-optimization',
						},
					],
				},
				{
					label: 'API Reference',
					items: [
						{
							label: 'Core',
							items: [{ autogenerate: { directory: 'api/core' } }],
						},
					],
				},
				{
					label: 'Troubleshooting',
					items: [
						{
							label: 'Training Integrations',
							items: [
								{ label: 'slime', slug: 'troubleshooting/slime' },
							],
						},
					],
				},
			],
		}),
	],
});
