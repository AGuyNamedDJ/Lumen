const { client } = require("../index");

// Create Conversation
async function createConversation({ userId }) {
    try {
        const result = await client.query(`
            INSERT INTO conversations (user_id)
            VALUES ($1)
            RETURNING *;
        `, [userId]);

        return result.rows[0];
    } catch (error) {
        console.error('Error creating conversation:', error);
        throw error;
    }
}

// Get All Conversations
async function getAllConversations() {
    try {
        const result = await client.query('SELECT * FROM conversations;');
        return result.rows;
    } catch (error) {
        console.error('Error getting all conversations:', error);
        throw error;
    }
}

// Get Conversation By ID
async function getConversationById(id) {
    try {
        const result = await client.query('SELECT * FROM conversations WHERE id = $1;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error getting conversation by ID:', error);
        throw error;
    }
}

// Update Conversation
async function updateConversation(id, fields = {}) {
    const setString = Object.keys(fields).map((key, index) => `"${key}"=$${index + 1}`).join(', ');

    try {
        const result = await client.query(`
            UPDATE conversations
            SET ${setString}
            WHERE id = $${Object.keys(fields).length + 1}
            RETURNING *;
        `, [...Object.values(fields), id]);

        return result.rows[0];
    } catch (error) {
        console.error('Error updating conversation:', error);
        throw error;
    }
}

// Delete Conversation
async function deleteConversation(id) {
    try {
        const result = await client.query('DELETE FROM conversations WHERE id = $1 RETURNING *;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error deleting conversation:', error);
        throw error;
    }
}

module.exports = {
    createConversation,
    getAllConversations,
    getConversationById,
    updateConversation,
    deleteConversation
}